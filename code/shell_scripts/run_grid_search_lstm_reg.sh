#!/bin/sh
for mode in rand static non-static
do
    echo ${mode}
    for lr in 0.001 0.01 0.1
    do
        for dropout in 0.4 0.5 0.6
        do
            for num_layers in 1 2
            do
                    python -u -m models.reg_lstm --dataset CongressionalHearing --mode ${mode} --bidirectional --num-layers ${num_layers} --hidden-dim 512 --wdrop 0.1 --embed-droprate 0.2 --beta-ema 0.99 --dropout ${dropout} --evaluate-dev --patience 5 --lr ${lr}  --epochs 30 --seed 1234 --metrics-json metrics_reg_lstm_pat5_mode${mode}_lr${lr}_numlayers${num_layers}_dropout${dropout}_1.json > ch_reg_lstm_pat5_mode${mode}_lr${lr}_numlayers${num_layers}_dropout${dropout}_1.log 2>&1
                    python -u -m models.reg_lstm --dataset CongressionalHearing --mode ${mode} --bidirectional --num-layers ${num_layers} --hidden-dim 512 --wdrop 0.1 --embed-droprate 0.2 --beta-ema 0.99 --dropout ${dropout} --evaluate-dev --patience 5 --lr ${lr}  --epochs 30 --seed 5678 --metrics-json metrics_reg_lstm_pat5_mode${mode}_lr${lr}_numlayers${num_layers}_dropout${dropout}_2.json > ch_reg_lstm_pat5_mode${mode}_lr${lr}_numlayers${num_layers}_dropout${dropout}_2.log 2>&1
                    python -u -m models.reg_lstm --dataset CongressionalHearing --mode ${mode} --bidirectional --num-layers ${num_layers} --hidden-dim 512 --wdrop 0.1 --embed-droprate 0.2 --beta-ema 0.99 --dropout ${dropout} --evaluate-dev --patience 5 --lr ${lr}  --epochs 30 --seed 9012 --metrics-json metrics_reg_lstm_pat5_mode${mode}_lr${lr}_numlayers${num_layers}_dropout${dropout}_3.json > ch_reg_lstm_pat5_mode${mode}_lr${lr}_numlayers${num_layers}_dropout${dropout}_3.log 2>&1
            done
        done
    done
done

