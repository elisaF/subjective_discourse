import argparse
import csv
import json
import numpy as np
import os
from common import utils

from collections import defaultdict


def process_json_results(json_dir, save_file, split, label_suffix):
    results_dict = {}

    # get model names
    model_names = []
    json_pattern = '*.json_' + split
    json_files = utils.get_files_with_pattern(json_dir, json_pattern)
    for json_file in json_files:
        ext_loc = json_file.index('.json')
        model_names.append('_'.join(json_file[:ext_loc].split('_')[:-1]))

    for model_name in set(model_names):
        file_pattern = '_'.join([model_name, '*.json', split])
        json_files = utils.get_files_with_pattern(json_dir, file_pattern)
        json_files = [os.path.join(json_dir, json_file) for json_file in json_files]
        results_dict[model_name] = process_model_results(json_files, label_suffix)
    with open(save_file, 'w') as f:
        header_row = ['model']
        header_row.extend(list(results_dict.values())[0].keys())
        w = csv.writer(f, delimiter='\t')
        w.writerow(header_row)
        for model_name, results in results_dict.items():
            row = [model_name]
            row.extend(results.values())
            w.writerow(row)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Json Results Processor 1.0')
    parser.add_argument('--json_dir', type=str, help='Directory with json files to process', default='results')
    parser.add_argument('--save_file', type=str, help='File to save results', default='results')
    parser.add_argument('--split', type=str, help='Data split to process', default='dev')
    parser.add_argument('--label_suffix', type=str, help='Suffix for metric labels', default='')

    args = parser.parse_args()
    print("Parser args: ", args)

    process_json_results(args.json_dir, args.save_file, args.split, args.label_suffix)


# python -m analyses.process_results --json_dir /Users/elisa/Documents/CompLing/congressional_hearing/results/roberta_question/ --save_file /Users/elisa/Documents/CompLing/congressional_hearing/results/roberta_question/roberta_question_summary.tsv