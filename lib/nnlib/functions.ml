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
  let rec loop y_true y_pred total_loss =
    match y_true with
    | [] -> total_loss
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
    | [] -> grad_output
    | y_true_hd :: _ ->
      let y_pred_hd = List.hd y_pred in
      let y_pred_hd_softmax = softmax y_pred_hd in
      let grad = N.sub y_pred_hd_softmax y_true_hd in
      loop (List.tl y_true) (List.tl y_pred) (grad :: grad_output)
  in
  loop y_true y_pred []
;;

let tanh_grad x = N.(scalar_sub 1. (mul x x))

let one_hot_encode x num_classes =
  let one_hot_matrix = N.zeros [| (N.shape x).(0); num_classes |] in
  let rec loop matrix i =
    if i < (N.shape x).(0)
    then (
      let idx = int_of_float (N.get x [| i; 0 |]) in
      N.set matrix [| i; idx |] 1.;
      loop matrix (i + 1))
    else matrix
  in
  loop one_hot_matrix 0
;;
