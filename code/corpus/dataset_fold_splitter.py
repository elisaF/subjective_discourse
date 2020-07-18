import pandas as pd
import os
import random
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from sklearn.model_selection import GroupKFold

COLUMN_NAMES = ['qa_index_digits', 'gold_labels_binary', 'r_text', 'q_text', 'q_text_last_question',
                'gold_sentiments', 'gold_q_sentiments', 'gold_workers', 'gold_q_intents', 'hit_order',  # 5
                'q_speaker_role', 'r_speaker_role', 'gold_q_intents_num', 'q_speaker_party', #10
                'gold_sentiments_num', 'gold_q_sentiments_num', 'gold_sentiments_coarse_num',  # 14
                'q_speaker',  # 17
                'gold_sentiments_binary', 'gold_q_sentiments_count', 'gold_sentiments_coarse_count',  # 18
                'gold_sentiments_coarse_binary',  # 21
                'gold_q_sentiments_coarse_count', 'gold_q_sentiments_coarse_num', 'gold_q_sentiments_coarse_binary',  # 22
                'entropy', 'entropy_norm', 'entropy_norm_buckets', 'entropy_binarized', 'question_type_num',  # 25
                'q_text_all_questions', 'gold_label_powerset', 'gold_label_powerset_binary']  # 'mace_entropy']  # 30


def get_args():
    parser = ArgumentParser(description="Split dataset into K folds")

    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--input_file', default=None)
    parser.add_argument('--output_dir', default=os.path.join(os.pardir, os.pardir, 'data', 'gold'))

    args = parser.parse_args()
    return args


def split_data(input_file, output_dir, seed, n_folds):
    df = pd.read_csv(input_file, sep='\t', dtype=str)

    # shuffle rows of dataframe several times
    for _ in range(5):
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # get group indeces
    hearing_to_num = {}
    for idx, hearing_id in enumerate(df['hearing_id'].unique()):
        hearing_to_num[hearing_id] = idx
    df['hearing_num'] = df['hearing_id'].map(hearing_to_num)
    group_idxs = df['hearing_num'].values

    outer_cv = GroupKFold(n_splits=n_folds)
    # Split X and y into K-partitions to outer CV
    indeces = df.index.values
    for (i, (train_index, test_index)) in enumerate(outer_cv.split(indeces, indeces, groups=group_idxs)):
        print('Fold: ', str(i), '/', str(outer_cv.get_n_splits()-1))
        fold_dir = os.path.join(output_dir, 'fold'+str(i))
        Path(fold_dir).mkdir(parents=True, exist_ok=True)
        file_name_train = os.path.join(fold_dir, 'train.tsv')
        df.loc[train_index][COLUMN_NAMES].to_csv(file_name_train, sep='\t', index=False)

        file_name_test = os.path.join(fold_dir, 'test.tsv')
        df.loc[test_index][COLUMN_NAMES].to_csv(file_name_test, sep='\t', index=False)


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    print('Args: ', args)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    split_data(args.input_file,  args.output_dir, args.seed, args.n_folds)

    #dataset_fold_splitter.py --input_file /Users/elisa/Documents/CompLing/congressional_hearing/data/gold/gold_with_features_05-18.tsv --output_dir /Users/elisa/Documents/CompLing/congressional_hearing/data/splits_folds --n_folds 5