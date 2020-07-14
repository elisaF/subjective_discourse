#!/bin/sh
for lr in 3e-5
do
    echo ${lr}
    for warmup in 0.1
    do
        for decay in 0.1
        do
            for batch_size in 8
            do
            python -u -m models.bert --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-dev --patience 30 --lr ${lr} --warmup-proportion ${warmup} --weight-decay ${decay} --batch-size ${batch_size} --epochs 30 --seed 1234 --metrics-json metrics_roberta_pat30_lr${lr}_warmup${warmup}_decay${decay}_batch${batch_size}_1.json > ch_roberta_pat30_lr${lr}_warmup${warmup}_decay${decay}_batch${batch_size}_1.log 2>&1
            python -u -m models.bert --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-dev --patience 30 --lr ${lr} --warmup-proportion ${warmup} --weight-decay ${decay} --batch-size ${batch_size} --epochs 30 --seed 5678 --metrics-json metrics_roberta_pat30_lr${lr}_warmup${warmup}_decay${decay}_batch${batch_size}_2.json > ch_roberta_pat30_lr${lr}_warmup${warmup}_decay${decay}_batch${batch_size}_2.log 2>&1
            python -u -m models.bert --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-dev --patience 30 --lr ${lr} --warmup-proportion ${warmup} --weight-decay ${decay} --batch-size ${batch_size} --epochs 30 --seed 9012 --metrics-json metrics_roberta_pat30_lr${lr}_warmup${warmup}_decay${decay}_batch${batch_size}_3.json > ch_roberta_pat30_lr${lr}_warmup${warmup}_decay${decay}_batch${batch_size}_3.log 2>&1
            done
        done
    done
done


