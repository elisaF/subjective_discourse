#!/bin/sh
for mode in rand static non-static
do
    echo ${mode}
    for lr in 0.0001 0.001 0.01
    do
        for dropout in 0.4 0.5 0.6
        do
            for weight_decay in 0 0.0001 0.001
            do
                    python -u -m models.fasttext --dataset CongressionalHearing --mode ${mode} --dropout ${dropout} --evaluate-dev --patience 5 --lr ${lr} --weight-decay ${weight_decay} --epochs 30 --seed 1234 --metrics-json metrics_fasttext_pat5_mode${mode}_lr${lr}_weightdecay${weight_decay}_dropout${dropout}_1.json > ch_fasttext_pat5__mode${mode}_lr${lr}_weightdecay${weight_decay}_dropout${dropout}_1.log 2>&1
                    python -u -m models.fasttext --dataset CongressionalHearing --mode ${mode} --dropout ${dropout} --evaluate-dev --patience 5 --lr ${lr} --weight-decay ${weight_decay} --epochs 30 --seed 5678 --metrics-json metrics_fasttext_pat5_mode${mode}_lr${lr}_weightdecay${weight_decay}_dropout${dropout}_2.json > ch_fasttext_pat5__mode${mode}_lr${lr}_weightdecay${weight_decay}_dropout${dropout}_2.log 2>&1
                    python -u -m models.fasttext --dataset CongressionalHearing --mode ${mode} --dropout ${dropout} --evaluate-dev --patience 5 --lr ${lr} --weight-decay ${weight_decay} --epochs 30 --seed 9012 --metrics-json metrics_fasttext_pat5_mode${mode}_lr${lr}_weightdecay${weight_decay}_dropout${dropout}_3.json > ch_fasttext_pat5__mode${mode}_lr${lr}_weightdecay${weight_decay}_dropout${dropout}_3.log 2>&1
            done
        done
    done
done

