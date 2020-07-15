import os

# String templates for logging results
LOG_HEADER_CLASS = 'Split  Acc.  Pr.  Re.   F1   Loss'
LOG_TEMPLATE_CLASS = ' '.join('{:>5s},{:>9.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))
LOG_HEADER_REG = 'Split  RMSE  Kendall Pears.  Spear.  Pears_Spear   Loss'
LOG_TEMPLATE_REG = ' '.join('{:>5s},{:8.4f},{:8.4f},{:8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))

METRIC_RMSE = 'RMSE'
METRIC_F1_MACRO = 'F1_MACRO'
METRIC_F1_BINARY = 'F1_BINARY'
METRIC_KENDALL = 'KENDALL'
METRIC_PEARSON = 'PEARSON'
METRIC_SPEARMAN = 'SPEARMAN'
METRIC_PEARSON_SPEARMAN = 'PEARSON_SPEARMAN'

TASK_CLASSIFICATION = 'classification'
TASK_REGRESSION = 'regression'

