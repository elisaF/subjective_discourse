#!/bin/bash
expnames=(r_text_gold_sentiments_coarse_num r_text_gold_sentiments_num r_text_q_text r_text_q_text_last_question r_text_hit_order r_text_q_speaker_role r_text_r_speaker_role r_text_gold_q_intents_num r_text_q_speaker_party r_text_gold_q_sentiments_num r_text_q_speaker r_text_gold_q_sentiments_coarse_num r_text_question_type_num r_text_q_text_all_questions r_text_q_text_last_2_sents r_text_q_text_last_3_sents r_text_q_text_first_question_and_rest r_text_q_text_last_question_and_rest)
declare -A column=(
  [r_text_gold_sentiments_num]=14
  [r_text_gold_sentiments_coarse_num]=16
  [r_text_q_text]=3
  [r_text_q_text_last_question]=4
  [r_text_hit_order]=9
  [r_text_q_speaker_role]=10
  [r_text_r_speaker_role]=11
  [r_text_gold_q_intents_num]=12
  [r_text_q_speaker_party]=13
  [r_text_gold_q_sentiments_num]=15
  [r_text_q_speaker]=17
  [r_text_gold_q_sentiments_coarse_num]=23
  [r_text_question_type_num]=29
  [r_text_q_text_all_questions]=30
  [r_text_q_text_last_2_sents]=33
  [r_text_q_text_last_3_sents]=34
  [r_text_q_text_first_question_and_rest]=35
  [r_text_q_text_last_question_and_rest]=36
)

for expname in "${expnames[@]}"; do
  echo $expname
  python -u -m models.bert.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 5 --lr 1e-5 --warmup-proportion 0.01 --weight-decay 0 --batch-size 8 --epochs 30 --seed 7890 --metrics-json metrics_roberta_classification_${expname}_test.json --first-input-column 2 --use-second-input --second-input-column ${column[$expname]} > ch_roberta_classification_${expname}_test.log 2>&1
done


