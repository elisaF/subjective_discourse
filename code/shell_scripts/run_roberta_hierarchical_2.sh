#!/bin/bash
expnames=(r_text_q_text_last_question r_text_gold_sentiments_coarse_num r_text_gold_sentiments_num)
declare -A column=(
  [r_text_q_text_last_question]=4
  [r_text_gold_sentiments_coarse_num]=16
  [r_text_gold_sentiments_num]=14
)

for expname in "${expnames[@]}"; do
  echo $expname
  python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hierarchical_${expname}_test.json --first-input-column 2 --use-second-input --second-input-column ${column[$expname]}  > ch_roberta_hierarchical_${expname}_test.log 2>&1
done


