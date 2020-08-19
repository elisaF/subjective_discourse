#!/bin/bash
expnames=(r_text)
declare -A column=(
  [r_text]=2
  [quest]=3
  [last_quest]=4
)
count=1
 for seed in 7890 5678 2345
 do
    echo ${count}
    for expname in "${expnames[@]}"; do
      echo $expname
      python -u -m models.bert_hier --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-dev --patience 10 --lr 1e-5 --warmup-proportion 0.01 --weight-decay 0 --batch-size 8 --epochs 60 --seed ${seed} --metrics-json metrics_roberta_hierarchical_${expname}_dev_${count}.json --first-input-column ${column[$expname]}  > ch_roberta_hierarchical_${expname}_dev_${count}.log 2>&1
      count=$((count + 1))
    done
done

