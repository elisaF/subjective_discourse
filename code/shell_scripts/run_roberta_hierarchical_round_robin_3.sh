#!/bin/bash
expnames=(gold_sentiments_num q_text_first_question_and_rest r_text)
declare -A column=(
     [gold_sentiments_num]=14
     [gold_q_sentiments_coarse_num]=23
     [r_text]=2
     [q_text_last_question]=4
     [q_text_first_question_and_rest]=35
 )

for first in "${expnames[@]}"; do
     for second in "${expnames[@]}"; do
         for third in  "${expnames[@]}"; do
               if [[ "$first" != "$second" ]] && [[ "$second" != "$third" ]] && [[ "$first" != "$third" ]]; then
                   echo $first $second $third
                   python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 1e-5 --warmup-proportion 0.01 --weight-decay 0 --batch-size 8 --epochs 30 --seed 7890 --metrics-json metrics_roberta_hierarchical_${first}_${second}_${third}_test.json --first-input-column ${column[$first]} --use-second-input --second-input-column ${column[$second]} --use-third-input --third-input-column ${column[$third]} > ch_roberta_hierarchical_${first}_${second}_${third}_test.log 2>&1
               fi
         done
     done
done
