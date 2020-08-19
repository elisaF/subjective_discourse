# Subjective Acts and Intents
This repo contains the code and data for analyzing subjective judgments of witness responses in U.S. congressional hearings (link paper here). If you make use of the data or code, please cite:

`
Ferracane, Elisa TBD
`

## Dataset
If you're here just for the data, you can download 

## Code
### Setup:
First, create a conda environment and activate it:
```
conda create --name subjective python=3.8
conda activate subjective
```

Install pytorch and cuda:
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Clone this repo and install the requirements:
```
git clone https://github.com/elisaF/subjective_discourse
cd subjective_discourse
pip install -r requirements.txt
```

Unpack the data splits:
```
cd subjective_discourse/data/gold
tar -zxvf gold_cv_dev_data.tar.gz
```

### Classification Task:
The multi-label classification task consists of predicting all the possible response labels, and is evaluated with macro-averaged F1.

**Roberta:** predict the response labels using the response text.

```shell
python -u -m models.bert.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 5 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_classification_r_text_test.json --first-input-column 2  > ch_roberta_classification_r_text_test.log 2>&1
```

**Hierarchical:** predict the response labels while also training to predict the conversation acts.

```shell
python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hierarchical_r_text_test.json --first-input-column 2 > ch_roberta_hierarchical_r_text_test.log 2>&1 &
```

**+Question:** predict the response labels as in the hierarchical model, but additionally using the last question.
```shell
python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hierarchical_q_text_last_question_r_text_test.json --first-input-column 4 --use-second-input --second-input-column 2  > ch_roberta_hierarchical_q_text_last_question_r_text_test.log 2>&1
```

**+Annotator:** predict the response labels as in the hierarchical model, but additionally using the annotator sentiments.
```shell
python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hierarchical_gold_sentiments_coarse_num_r_text_test.json --first-input-column 16 --use-second-input --second-input-column 2  > ch_roberta_hierarchical_gold_sentiments_coarse_num_r_text_test.log 2>&1
```

**+Question+Annotator:** predict the response labels as in the hierarchical model, but additionally using the last question and the annotator sentiments.
```shell
python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hierarchical_r_text_gold_sentiments_coarse_num_q_text_last_question_test.json --first-input-column 2 --use-second-input --second-input-column 16  --use-third-input --third-input-column 4 > ch_roberta_hierarchical_r_text_gold_sentiments_coarse_num_q_text_last_question__test.log 2>&1
```
### Regression Task:
The regression task consists of predicting the normalized entropy of the response label distribution, and is evaluated with RMSE.

**Roberta:** predict the response labels using only the response text. Note this experimental model is run on the dev fold
```
cd subjective_discourse/code/models/hedwig
../../shell_scripts/run_roberta_regression_dev.sh
```
