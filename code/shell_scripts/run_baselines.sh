#!/bin/sh
count=1
for seed in 1234 5678 9012
do
    echo ${count}
    python -u -m models.kim_cnn --mode static --dataset CongressionalHearing --batch-size 32 --lr 0.01 --epochs 30 --dropout 0.5 --seed ${seed} --metrics-json metrics_kimcnn_${count}.json --evaluate-dev > ch_kimcnn_${count}_dev.log 2>&1
    python -u -m models.fasttext --dataset CongressionalHearing --seed ${seed} --metrics-json metrics_fasttext_${count}.json --mode rand --evaluate-dev > ch_fasttext_${count}_dev.log 2>&1
    python -u -m models.han --dataset CongressionalHearing --mode static --batch-size 32 --lr 0.01 --epochs 30 --seed ${seed} --metrics-json metrics_han_${count}.json --evaluate-dev > ch_han_${count}_dev.log 2>&1
    python -u -m models.lr --dataset CongressionalHearing --max-vocab-size 3750 --seed ${seed} --metrics-json metrics_lr_${count}.json --evaluate-dev > ch_lr_${count}_dev.log 2>&1
    python -u -m models.reg_lstm --dataset CongressionalHearing --mode static --batch-size 32 --lr 0.01 --epochs 30 --bidirectional --num-layers 1 --hidden-dim 512 --wdrop 0.1 --embed-droprate 0.2 --dropout 0.5 --beta-ema 0.99 --seed ${seed} --metrics-json metrics_lstm_${count}.json --evaluate-dev > ch_lstm_${count}_dev.log 2>&1
    count=$((count + 1))
done

