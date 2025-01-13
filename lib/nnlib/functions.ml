module N = Owl_base_dense_ndarray.D

let mse_loss y_true y_pred =
  let diff = N.sub y_true y_pred in
  let square = N.mul diff diff in
  N.sum' square /. Float.of_int (N.shape y_true).(0)
;;

let mse_loss_grad y_true y_pred =
  let diff = N.sub y_pred y_true in
  N.mul_scalar diff (2. /. Float.of_int (N.shape y_true).(0))
;;

let cross_entropy_loss y_true y_pred =
  let y_pred = N.clip_by_value y_pred ~amin:1e-7 ~amax:1. in
  let y_pred = N.log y_pred in
  let y_true = N.mul_scalar y_true (-1.) in
  let sum = N.sum' (N.mul y_true y_pred) in
  (* scale by batch_size *)
  sum /. Float.of_int (N.shape y_true).(0)
;;

let cross_entropy_loss_grad y_true y_pred =
  let y_pred = N.clip_by_value y_pred ~amin:1e-7 ~amax:1. in
  let y_true = N.mul_scalar y_true (-1.) in
  let grad = N.div y_true y_pred in
  N.div_scalar grad (Float.of_int (N.shape y_true).(0))
;;

(* row-wise softmax *)
let softmax m =
  let row_max = N.max ~axis:1 m in
  (* avoid exp overflow *)
  let shifted_m = N.sub m row_max in
  let exp_m = N.map Float.exp shifted_m in
  let row_sum = N.sum ~axis:1 exp_m in
  N.div exp_m row_sum
;;

let softmax_ce_loss y_true y_pred =
  assert (List.length y_pred = List.length y_true);
  let len = List.length y_true in
  let rec loop y_true y_pred total_loss =
    match y_true with
    | [] -> total_loss /. Float.of_int len
    | y_true_hd :: _ ->
      let y_pred_hd = List.hd y_pred in
      let y_pred_hd_softmax = softmax y_pred_hd in
      let loss = cross_entropy_loss y_true_hd y_pred_hd_softmax in
      loop (List.tl y_true) (List.tl y_pred) (total_loss +. loss)
  in
  loop y_true y_pred 0.
;;

let softmax_ce_loss_grad y_true y_pred =
  assert (List.length y_pred = List.length y_true);
  let rec loop y_true y_pred grad_output =
    match y_true with
    | [] -> List.rev grad_output
    | y_true_hd :: _ ->
      let y_pred_hd = List.hd y_pred in
      let y_pred_hd_softmax = softmax y_pred_hd in
      let grad = N.sub y_pred_hd_softmax y_true_hd in
      loop (List.tl y_true) (List.tl y_pred) (grad :: grad_output)
  in
  loop y_true y_pred []
;;

let clip_by_value g threshold =
  N.map
    (fun x ->
      if x > threshold
      then threshold
      else if x < -1. *. threshold
      then -1. *. threshold
      else x)
    g
;;

let clip_by_norm g threshold =
  let norm = N.l2norm' g in
  if norm > threshold then N.mul_scalar g (threshold /. norm) else g
;;

let tanh_grad x = N.(scalar_sub 1. (mul x x))

let one_hot_encode x num_classes =
  let one_hot_matrix = N.zeros [| (N.shape x).(0); num_classes |] in
  let rec loop matrix i =
    if i = (N.shape x).(0)
    then matrix
    else (
      let idx = int_of_float (N.get x [| i; 0 |]) in
      N.set matrix [| i; idx |] 1.;
      loop matrix (i + 1))
  in
  loop one_hot_matrix 0
;;

let float_eq a b = abs_float (a -. b) < 0.0001

let one_hot_max arr =
  let max_value = N.max' arr in
  let result = N.zeros (N.shape arr) in
  let result = N.map (fun x -> if float_eq x max_value then 1.0 else 0.0) result in
  result
;;

let argmax_1d arr =
  let init_max = Float.neg_infinity in
  let rec find_max idx current_max current_idx =
    if idx = N.numel arr
    then current_idx
    else (
      let value = N.get arr [| idx |] in
      if value > current_max
      then find_max (idx + 1) value idx
      else find_max (idx + 1) current_max current_idx)
  in
  find_max 1 init_max 0
;;
