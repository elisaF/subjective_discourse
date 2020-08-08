#!/bin/bash
expnames=(q_text_last_question_r_text_q_speaker_role_gold_sentiments_coarse_num q_text_last_question_r_text_q_speaker_role_gold_sentiments_num)
declare -A column=(
  [q_text_last_question_r_text_q_speaker_role_gold_sentiments_coarse_num]=16
  [q_text_last_question_q_speaker_role_r_text]=2
  [q_text_last_question_r_text_q_speaker_role_gold_sentiments_num]=14
  []=2
)

for expname in "${expnames[@]}"; do
  echo $expname
  python -u -m models.bert.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 5 --lr 1e-5 --warmup-proportion 0.01 --weight-decay 0 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_classification_${expname}_test.json --second-input-column 2 --use-second-input --first-input-column 4 --use-third-input --third-input-column 10 --use-fourth-input --fourth-input-column ${column[$expname]} > ch_roberta_classification_${expname}_test.log 2>&1
done


