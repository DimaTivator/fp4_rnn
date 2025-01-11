[@@@ocaml.warning "-32"]

open Nnlib

(* open Str *)
(* open Utils *)
(* open Camomile *)
module Case_Map = Camomile.CaseMap.Make (Camomile.UTF8)
module N = Owl_base_dense_ndarray.D

let run_epoch ~model ~batches ~lr =
  let num_batches = List.length batches in
  let rec loop model batches total_loss =
    match batches with
    | [] -> model, total_loss /. Float.of_int num_batches
    | batch :: batches_tl ->
      let batch_x, batch_y = batch in
      let outputs, hidden_states = Nn.RNN.forward model batch_x in
      let loss = Functions.softmax_ce_loss batch_y outputs in
      let grad_output = Functions.softmax_ce_loss_grad batch_y outputs in
      let grad = Nn.RNN.backward model batch_x grad_output hidden_states in
      let model = Nn.RNN.update model grad lr in
      loop model batches_tl (total_loss +. loss)
  in
  loop model batches 0.
;;

let train ~model ~batches ~epochs ~lr =
  let rec loop model epoch lr =
    if epoch = epochs
    then model
    else (
      let model, loss = run_epoch ~model ~batches ~lr in
      let lr = if (epoch + 1) mod 5 = 0 then lr *. 0.3 else lr in
      Stdio.prerr_endline (Float.to_string loss);
      loop model (epoch + 1) lr)
  in
  loop model 0 lr
;;

(* let words = List.map Case_Map.lowercase (Utils.read_words_from_file "data/names.txt");; *)
let words = Utils.read_words_from_file "data/names.txt";;
let letters = List.flatten (List.map Utils.split_into_letters words);;
let vocab = List.sort_uniq compare letters;;
(* List.iter (Printf.printf "%d ") vocab;; *)
let vocab_size = List.length vocab
let letter_to_token = List.mapi (fun idx letter -> letter, idx) vocab
let token_to_letter = List.mapi (fun idx letter -> idx, letter) vocab

let tokens =
  List.map
    (fun letter ->
      try List.assoc letter letter_to_token with
      | Not_found -> 1)
    letters
;;

let float_tokens = List.map Float.of_int tokens
let batch_size = 256
let seq_length = 20
let batches = Utils.get_batches float_tokens batch_size seq_length vocab_size
let cfg : Nn.RNN.config = { vocab_size; hidden_size = 300 }
let model = Nn.RNN.create cfg;;

prerr_endline "-----------------Start training-----------------";;
let model = train ~model ~batches ~epochs:25 ~lr:0.01;;
prerr_endline "-----------------End training-----------------"

let () =
  let to_arr x = N.of_array [| x |] [| 1; 1 |] in
  let input = List.map to_arr [ 6.; 38.; 42.; 30.; 0. ] in
  let res_tokens = Nn.RNN.generate model input 30 in
  let tokens_list = List.map (fun x -> N.get x [| 0; 0 |]) res_tokens in
  let letters =
    List.map (fun x -> List.assoc (Int.of_float x) token_to_letter) tokens_list
  in
  List.iter (Printf.printf "%d ") letters;
;;

