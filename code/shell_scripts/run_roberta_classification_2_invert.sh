#!/bin/bash
expnames=(gold_sentiments_coarse_num_r_text gold_sentiments_num_r_text q_text_r_text q_text_last_question_r_text hit_order_r_text q_speaker_role_r_text r_speaker_role_r_text gold_q_intents_num_r_text q_speaker_party_r_text gold_q_sentiments_num_r_text q_speaker_r_text gold_q_sentiments_coarse_num_r_text question_type_num_r_text q_text_all_questions_r_text)
declare -A column=(
  [gold_sentiments_num_r_text]=14
  [gold_sentiments_coarse_num_r_text]=16
  [q_text_r_text]=3
  [q_text_last_question_r_text]=4
  [hit_order_r_text]=9
  [q_speaker_role_r_text]=10
  [r_speaker_role_r_text]=11
  [gold_q_intents_num_r_text]=12
  [q_speaker_party_r_text]=13
  [gold_q_sentiments_num_r_text]=15
  [q_speaker_r_text]=17
  [gold_q_sentiments_coarse_num_r_text]=23
  [question_type_num_r_text]=29
  [q_text_all_questions_r_text]=30
)

for expname in "${expnames[@]}"; do
  echo $expname
  python -u -m models.bert.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 5 --lr 1e-5 --warmup-proportion 0.1 --weight-decay 0.001 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_classification_${expname}_test.json --second-input-column 2 --use-second-input --first-input-column ${column[$expname]} > ch_roberta_classification_${expname}_test.log 2>&1
done


