#!/bin/bash
expnames=(r_text_q_text_last_question_gold_sentiments_num r_text_q_text_last_question_gold_sentiments_coarse_num)
declare -A column=(
  [r_text_q_text_last_question_gold_sentiments_num]=14
  [r_text_q_text_last_question_gold_sentiments_coarse_num]=16
)

for expname in "${expnames[@]}"; do
  echo $expname
  python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 1e-5 --warmup-proportion 0.01 --weight-decay 0 --batch-size 8 --epochs 30 --seed 7890 --metrics-json metrics_roberta_hierarchical_${expname}_test.json --first-input-column 2 --use-second-input --second-input-column 4 --use-third-input --third-input-column ${column[$expname]}  > ch_roberta_hierarchical_${expname}_test.log 2>&1
done


