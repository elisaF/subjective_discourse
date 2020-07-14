#!/bin/sh
for lr in 1e-5 2e-5 3e-5
do
    echo ${lr}
    for warmup in 0.001 0.01 0.1
    do
        for decay in 0.001 0.01 0.1
        do
            python -u -m models.bert --dataset CongressionalHearing --model-family albert --model albert-base-v2 --max-seq-length 512 --evaluate-dev --patience 5 --lr ${lr} --warmup-proportion ${warmup} --weight-decay ${decay} --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_albert_pat5_lr${lr}_warmup${warmup}_decay${decay}_1.json > ch_albert_pat5_lr${lr}_warmup${warmup}_decay${decay}_1.log 2>&1
            python -u -m models.bert --dataset CongressionalHearing --model-family albert --model albert-base-v2 --max-seq-length 512 --evaluate-dev --patience 5 --lr ${lr} --warmup-proportion ${warmup} --weight-decay ${decay} --batch-size 8 --epochs 30 --seed 5678 --metrics-json metrics_albert_pat5_lr${lr}_warmup${warmup}_decay${decay}_2.json > ch_albert_pat5_lr${lr}_warmup${warmup}_decay${decay}_2.log 2>&1
            python -u -m models.bert --dataset CongressionalHearing --model-family albert --model albert-base-v2 --max-seq-length 512 --evaluate-dev --patience 5 --lr ${lr} --warmup-proportion ${warmup} --weight-decay ${decay} --batch-size 8 --epochs 30 --seed 9012 --metrics-json metrics_albert_pat5_lr${lr}_warmup${warmup}_decay${decay}_3.json > ch_albert_pat5_lr${lr}_warmup${warmup}_decay${decay}_3.log 2>&1
        done
    done
done

