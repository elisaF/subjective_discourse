# Subjective Acts and Intents
This repo contains the code and data for analyzing subjective judgments of witness responses in U.S. congressional hearings (link paper here). If you make use of the data or code, please cite:

`
Ferracane, Elisa TBD
`

## Dataset
If you're here just for the data, you can download 

## Code
### Classification Task:
The multi-label classification task consists of predicting all the possible response labels, and is evaluated with macro-averaged F1.

**Roberta:** predict the response labels using the response text

**Hierarchical:** predict the response labels while also training to predict the conversation acts:

```shell
python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hierarchical_r_text_test.json --first-input-column 2 > ch_roberta_hierarchical_r_text_test.log 2>&1 &
```

**+Question+Annotator:** predict the response labels as in the hierarchical model, but additionally using the last question and the annotator sentiments:
```shell
```
### Regression Task:
The regression task consists of predicting the normalized entropy of the response label distribution, and is evaluated with RMSE.

**Roberta:** predict the response labels using only the response text:
