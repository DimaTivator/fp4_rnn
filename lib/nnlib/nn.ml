module N = Owl_base_dense_ndarray.D

module Linear = struct
  type t =
    { weight : N.arr
    ; bias : N.arr
    }

  type config =
    { input_size : int
    ; output_size : int
    }

  type grad =
    { grad_input : N.arr
    ; grad_weight : N.arr
    ; grad_bias : N.arr
    }

  let create config =
    let variance = Float.(1. /. sqrt (of_int config.input_size)) in
    let weight =
      N.gaussian
        ~mu:0.
        ~sigma:(Float.sqrt variance)
        [| config.input_size; config.output_size |]
    in
    let bias = N.zeros [| 1; config.output_size |] in
    { weight; bias }
  ;;

  let forward linear input = N.add linear.bias (N.dot input linear.weight)

  let backward linear input grad_output =
    let grad_input = N.dot grad_output (N.transpose linear.weight) in
    let grad_weight = N.dot (N.transpose input) grad_output in
    let grad_bias = N.sum ~axis:0 grad_output in
    { grad_input; grad_weight; grad_bias }
  ;;

  let update linear grad lr =
    let weight = N.sub linear.weight (N.mul_scalar grad.grad_weight lr) in
    let bias = N.sub linear.bias (N.mul_scalar grad.grad_bias lr) in
    { weight; bias }
  ;;
end

module RNN = struct
  type config =
    { vocab_size : int
    ; hidden_size : int
    }

  type t =
    { input_to_hidden : Linear.t
    ; hidden_to_hidden : Linear.t
    ; hidden_to_output : Linear.t
    ; cfg : config
    }

  type grad =
    { input_to_hidden : Linear.grad
    ; hidden_to_hidden : Linear.grad
    ; hidden_to_output : Linear.grad
    }

  let create cfg =
    let input_to_hidden =
      Linear.create { input_size = cfg.vocab_size; output_size = cfg.hidden_size }
    in
    let hidden_to_hidden =
      Linear.create { input_size = cfg.hidden_size; output_size = cfg.hidden_size }
    in
    let hidden_to_output =
      Linear.create { input_size = cfg.hidden_size; output_size = cfg.vocab_size }
    in
    { input_to_hidden; hidden_to_hidden; hidden_to_output; cfg }
  ;;

  let init_state batch_size hidden_size = N.zeros [| batch_size; hidden_size |]

  let check_empty_list = function
    | [] -> failwith "Empty list"
    | _ -> ()
  ;;

  let forward rnn input =
    check_empty_list input;
    let batch_size = (N.shape (List.hd input)).(0) in
    let hidden_states = [ init_state batch_size rnn.cfg.hidden_size ] in
    let rec process_sequence input hidden_states outputs =
      match input with
      | [] -> List.rev outputs, List.rev hidden_states
      | input_t :: tl ->
        let input_t = Functions.one_hot_encode input_t rnn.cfg.vocab_size in
        let hidden_state =
          let prev_hidden_state = List.hd hidden_states in
          let embedding = Linear.forward rnn.input_to_hidden input_t in
          let hidden_to_hidden_output =
            Linear.forward rnn.hidden_to_hidden prev_hidden_state
          in
          let combined = N.add embedding hidden_to_hidden_output in
          N.tanh combined
        in
        let output_t = Linear.forward rnn.hidden_to_output hidden_state in
        process_sequence tl (hidden_state :: hidden_states) (output_t :: outputs)
    in
    process_sequence input hidden_states []
  ;;

  let generate rnn input length =
    let rec loop rnn length output =
      if length = 0
      then output
      else (
        let model_output, _ = forward rnn output in
        let output_t = List.hd (List.rev model_output) in
        let output_t = Functions.softmax output_t in
        let output_t = N.reshape output_t [| rnn.cfg.vocab_size |] in
        let token = Float.of_int (Functions.argmax_1d output_t) in
        loop rnn (length - 1) (output @ [ N.of_array [| token |] [| 1; 1 |] ]))
    in
    loop rnn length input
  ;;

  let create_linear_grad input_shape output_shape =
    { Linear.grad_input = N.zeros input_shape
    ; Linear.grad_weight = N.zeros [| input_shape.(1); output_shape.(1) |]
    ; Linear.grad_bias = N.zeros [| 1; output_shape.(1) |]
    }
  ;;

  let init_grad cfg batch_size =
    let grad_input_to_hidden =
      create_linear_grad
        [| batch_size; cfg.vocab_size |]
        [| batch_size; cfg.hidden_size |]
    in
    let grad_hidden_to_hidden =
      create_linear_grad
        [| batch_size; cfg.hidden_size |]
        [| batch_size; cfg.hidden_size |]
    in
    let grad_hidden_to_output =
      create_linear_grad
        [| batch_size; cfg.hidden_size |]
        [| batch_size; cfg.vocab_size |]
    in
    { input_to_hidden = grad_input_to_hidden
    ; hidden_to_hidden = grad_hidden_to_hidden
    ; hidden_to_output = grad_hidden_to_output
    }
  ;;

  let add_grad grad1 grad2 =
    { Linear.grad_input = N.add grad1.Linear.grad_input grad2.Linear.grad_input
    ; Linear.grad_weight = N.add grad1.Linear.grad_weight grad2.Linear.grad_weight
    ; Linear.grad_bias = N.add grad1.Linear.grad_bias grad2.Linear.grad_bias
    }
  ;;

  let backward rnn input grad_output hidden_states =
    check_empty_list input;
    let input = List.rev input in
    let grad_output = List.rev grad_output in
    let hidden_states = List.rev hidden_states in
    let batch_size = (N.shape (List.hd input)).(0) in
    let grad = init_grad rnn.cfg batch_size in
    let grad_hidden_state_next = N.zeros [| batch_size; rnn.cfg.hidden_size |] in
    let rec backprop input grad grad_output grad_hidden_state_next hidden_states =
      match input with
      | [] -> grad
      | input_t :: tl ->
        let input_t = Functions.one_hot_encode input_t rnn.cfg.vocab_size in
        let hidden_state_t = List.hd hidden_states in
        let prev_hidden_state = List.nth hidden_states 1 in
        let grad_output_t = List.hd grad_output in
        let grad_hidden_to_output =
          Linear.backward rnn.hidden_to_output hidden_state_t grad_output_t
        in
        let grad_hidden_state =
          N.add
            grad_hidden_state_next
            (N.dot grad_output_t (N.transpose rnn.hidden_to_output.weight))
        in
        let grad_tanh = Functions.tanh_grad hidden_state_t in
        let grad_hidden_state_raw =
          N.mul grad_hidden_state grad_tanh
        in
        let grad_input_to_hidden =
          Linear.backward rnn.input_to_hidden input_t grad_hidden_state_raw
        in
        let grad_hidden_to_hidden =
          Linear.backward rnn.hidden_to_hidden prev_hidden_state grad_hidden_state_raw
        in
        let grad_hidden_state_next =
          N.dot grad_hidden_state_raw (N.transpose rnn.hidden_to_hidden.weight)
        in
        let grad =
          { input_to_hidden = add_grad grad.input_to_hidden grad_input_to_hidden
          ; hidden_to_hidden = add_grad grad.hidden_to_hidden grad_hidden_to_hidden
          ; hidden_to_output = add_grad grad.hidden_to_output grad_hidden_to_output
          }
        in
        backprop
          tl
          grad
          (List.tl grad_output)
          grad_hidden_state_next
          (List.tl hidden_states)
    in
    backprop input grad grad_output grad_hidden_state_next hidden_states
  ;;

  let update (rnn : t) grad lr =
    let clip_grad g =
      let threshold = 100. in
      let norm = N.l2norm' g in
      if norm > threshold then N.mul_scalar g (threshold /. norm) else g
    in
    let clip_linear_grad (g : Linear.grad) =
      { Linear.grad_input = clip_grad g.grad_input
      ; grad_weight = clip_grad g.grad_weight
      ; grad_bias = clip_grad g.grad_bias
      }
    in
    let grad =
      { input_to_hidden = clip_linear_grad grad.input_to_hidden
      ; hidden_to_hidden = clip_linear_grad grad.hidden_to_hidden
      ; hidden_to_output = clip_linear_grad grad.hidden_to_output
      }
    in
    let input_to_hidden = Linear.update rnn.input_to_hidden grad.input_to_hidden lr in
    let hidden_to_hidden = Linear.update rnn.hidden_to_hidden grad.hidden_to_hidden lr in
    let hidden_to_output = Linear.update rnn.hidden_to_output grad.hidden_to_output lr in
    { input_to_hidden; hidden_to_hidden; hidden_to_output; cfg = rnn.cfg }
  ;;
end
