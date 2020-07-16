#!/bin/sh
count=1
 for seed in 1234 5678 9012
 do
     echo ${count}
     python -u -m models.bert --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-dev --patience 5 --lr 5e-5 --warmup-proportion 0.1  --weight-decay 0.1  --batch-size 8 --epochs 30 --seed ${seed} --task regression --label-column 26 --evaluate-train --eval-metric RMSE --first-input-column 2 --metrics-json metrics_roberta_regression_dev_${count}.json > ch_roberta_regression_dev_${count}.log 2>&1
done