open Nnlib
open Camomile
module Case_Map = Camomile.CaseMap.Make (Camomile.UTF8)
module N = Owl_base_dense_ndarray.D

let print_list lst = List.iter (fun x -> N.print x) lst
let split_into_words str = Str.split (Str.regexp "[ \t\n\r,.;:!?()]+") str

let split_into_letters word =
  List.init (UTF8.length word) (fun i -> UChar.code (UTF8.get word i))
;;

let read_words_from_file filename =
  let ic = open_in filename in
  let rec read_lines acc =
    try
      let line = input_line ic in
      let words = split_into_words line in
      read_lines (words @ acc)
    with
    | End_of_file ->
      close_in ic;
      List.rev acc
  in
  read_lines []
;;

let rec take n lst =
  match n, lst with
  | 0, _ -> []
  | _, [] -> []
  | n, x :: xs -> x :: take (n - 1) xs
;;

let rec drop n lst =
  match n, lst with
  | 0, _ -> lst
  | _, [] -> []
  | n, _ :: xs -> drop (n - 1) xs
;;

let nzip lists =
  List.fold_left
    (fun accumulated_sequence new_sequence ->
      List.map2
        (fun current_list new_element -> current_list @ [ new_element ])
        accumulated_sequence
        new_sequence)
    (List.map (fun x -> [ x ]) (List.hd lists))
    (List.tl lists)
;;

let get_batches tokens batch_size seq_length vocab_size =
  let num_batches = List.length tokens / (batch_size * seq_length) in
  let rec get_sequenses n tokens sequenses =
    if n = batch_size
    then List.rev sequenses, tokens
    else (
      let seq = take seq_length tokens in
      let tokens = drop seq_length tokens in
      get_sequenses (n + 1) tokens (seq :: sequenses))
  in
  let rec loop n tokens batches =
    if n = num_batches
    then List.rev batches
    else (
      let sequences, tokens = get_sequenses 0 tokens [] in
      let nzipped_sequenses = nzip sequences in
      let batch_x =
        List.map
          (fun x -> N.of_array (Array.of_list x) [| batch_size; 1 |])
          nzipped_sequenses
      in
      let batch_y = List.tl batch_x in
      let batch_x = List.rev (List.tl (List.rev batch_x)) in
      let batch_y_one_hot =
        List.map (fun y -> Functions.one_hot_encode y vocab_size) batch_y
      in
      loop (n + 1) tokens ((batch_x, batch_y_one_hot) :: batches))
  in
  loop 0 tokens []
;;
