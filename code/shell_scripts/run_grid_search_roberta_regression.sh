#!/bin/sh
for lr in 5e-5 2e-5 2e-4 2e-3
do
    echo ${lr}
    for warmup in 0 0.01 0.1
    do
        for decay in 0 0.01 0.1
        do
            for batch_size in 8
            do
            python -u -m models.bert --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-dev --patience 5 --lr ${lr} --warmup-proportion ${warmup}  --weight-decay ${decay}  --batch-size ${batch_size} --epochs 30 --seed 1234 --task regression --label-column 26 --evaluate-train --eval-metric RMSE --metrics-json metrics_roberta_regression_rmse_pat5_lr${lr}_warmup${warmup}_decay${decay}_batch${batch_size}_1.json > ch_roberta_regression_rmse_pat5_lr${lr}_warmup${warmup}_decay${decay}_batch${batch_size}_1.log 2>&1
            python -u -m models.bert --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-dev --patience 5 --lr ${lr} --warmup-proportion ${warmup}  --weight-decay ${decay}  --batch-size ${batch_size} --epochs 30 --seed 5678 --task regression --label-column 26 --evaluate-train --eval-metric RMSE --metrics-json metrics_roberta_regression_rmse_pat5_lr${lr}_warmup${warmup}_decay${decay}_batch${batch_size}_2.json > ch_roberta_regression_rmse_pat5_lr${lr}_warmup${warmup}_decay${decay}_batch${batch_size}_2.log 2>&1
            python -u -m models.bert --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-dev --patience 5 --lr ${lr} --warmup-proportion ${warmup}  --weight-decay ${decay}  --batch-size ${batch_size} --epochs 30 --seed 9012 --task regression --label-column 26 --evaluate-train --eval-metric RMSE --metrics-json metrics_roberta_regression_rmse_pat5_lr${lr}_warmup${warmup}_decay${decay}_batch${batch_size}_3.json > ch_roberta_regression_rmse_pat5_lr${lr}_warmup${warmup}_decay${decay}_batch${batch_size}_3.log 2>&1
            done
        done
    done
done

