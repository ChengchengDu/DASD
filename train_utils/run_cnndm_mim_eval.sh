python run_eval.py  \
--model_name facebook/bart-large-cnn \
--data_dir /home/jazhan/data/PrekshaNema25/nema \
--model_path /home/jazhan/code/self_distill_qfs/no_distill_checkpoint/best_acc/ \
--save_path /home/jazhan/code/self_distill_qfs/no_distill_checkpoint/best_acc/ft_bart_nema_prediction.txt \
--reference_path /home/jazhan/data/PrekshaNema25/nema/test.target \
--score_path /home/jazhan/code/self_distill_qfs/no_distill_checkpoint/best_acc/ft_bart_nema_prediction_score.json \
--num_beams 6 \
--length_penalty 2.0 \
--max_length 20 \
--min_length 5 \
--eval_batch_size 16 \
--data_type nema \
--no_repeat_ngram_size 3 \
--early_stopping \
