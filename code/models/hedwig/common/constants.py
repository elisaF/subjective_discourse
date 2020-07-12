import os

# Model categories
BERT_MODELS = ['BERT-Base', 'BERT-Large', 'HBERT-Base', 'HBERT-Large']

# String templates for logging results
LOG_HEADER_CLASS = 'Split  Acc.  Pr.  Re.   F1   Loss'
LOG_TEMPLATE_CLASS = ' '.join('{:>5s},{:>9.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))
LOG_HEADER_REG = 'Split  RMSE  Kendall Pears.  Spear.  Pears_Spear   Loss'
LOG_TEMPLATE_REG = ' '.join('{:>5s},{:8.4f},{:8.4f},{:8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))

# Path to pretrained model and vocab files
MODEL_DATA_DIR = os.path.join(os.pardir, 'hedwig-data', 'models')
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-uncased'),
    'bert-large-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-large-uncased'),
    'bert-base-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-cased'),
    'bert-large-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-large-cased'),
    'bert-base-multilingual-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-uncased'),
    'bert-base-multilingual-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-cased')
}
PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-uncased-vocab.txt'),
    'bert-large-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-large-uncased-vocab.txt'),
    'bert-base-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-cased-vocab.txt'),
    'bert-large-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-large-cased-vocab.txt'),
    'bert-base-multilingual-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-uncased-vocab.txt'),
    'bert-base-multilingual-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-cased-vocab.txt')
}

METRIC_RMSE = 'RMSE'
METRIC_F1_MACRO = 'F1_MACRO'
METRIC_F1_BINARY = 'F1_BINARY'
METRIC_KENDALL = 'KENDALL'
METRIC_PEARSON = 'PEARSON'
METRIC_SPEARMAN = 'SPEARMAN'
METRIC_PEARSON_SPEARMAN = 'PEARSON_SPEARMAN'

TASK_CLASSIFICATION = 'classification'
TASK_REGRESSION = 'regression'
TASK_ORDINAL = 'ordinal'

