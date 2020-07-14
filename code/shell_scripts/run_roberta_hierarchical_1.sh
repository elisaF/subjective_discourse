#!/bin/bash
expnames=(r_text)
declare -A column=(
  [r_text]=2
  [quest]=3
  [last_quest]=4
)

for expname in "${expnames[@]}"; do
  echo $expname
  python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hierarchical_${expname}_test.json --first-input-column ${column[$expname]}  > ch_roberta_hierarchical_${expname}_test.log 2>&1
done


