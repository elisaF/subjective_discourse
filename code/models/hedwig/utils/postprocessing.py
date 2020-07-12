import csv
from collections import defaultdict
import csv
import json
import fnmatch
import numpy as np
import os


def get_files_with_pattern(dir_, pattern):
    matching_files = []
    for f in os.listdir(dir_):
        if fnmatch.fnmatch(f, pattern):
            matching_files.append(f)
    return matching_files


def process_json_results(json_prefix, save_file, split, label_suffix=''):
    json_pattern = json_prefix+'_fold*' + split + '*' + label_suffix
    json_files = get_files_with_pattern('.', json_pattern)
    results_dict = process_model_results(json_files, label_suffix)
    with open(save_file, 'w') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(results_dict.keys())
        w.writerow(results_dict.values())


def process_model_results(json_files, label_suffix):
    ignore_keys = ['support_class', 'confusion_matrix', 'label_set_info (id/gold/pred)', 'id_gold_pred']
    ignore_keys = [key+label_suffix for key in ignore_keys]

    # read all json files
    result_dicts = []
    for json_file in json_files:
        print('Reading json ', json_file)
        with open(json_file, 'r') as json_data:
            result_dicts.append(json.load(json_data))
    # merge dicts into one
    merged_dict = defaultdict(list)
    for d in result_dicts:
        for key, value in d.items():
            merged_dict[key].append(value)

    # calculate mean and variance
    summarized_dict = {}
    for key, value in merged_dict.items():
        if key not in ignore_keys:
            avg_key = key + '_avg'
            avg_value = np.mean(np.array(value), axis=0)
            summarized_dict[avg_key] = avg_value

            var_key = key + '_var'
            var_value = np.var(np.array(value), axis=0)
            summarized_dict[var_key] = var_value
    return summarized_dict
