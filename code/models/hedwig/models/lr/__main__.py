import json
import os
import random

import numpy as np
import torch.onnx
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from common.evaluators.bow_evaluator import BagOfWordsEvaluator
from common.trainers.bow_trainer import BagOfWordsTrainer
from datasets.bow_processors.aapd_processor import AAPDProcessor
from datasets.bow_processors.imdb_processor import IMDBProcessor
from datasets.bow_processors.reuters_processor import ReutersProcessor
from datasets.bow_processors.congressional_hearing_processor import CongressionalHearingProcessor
from datasets.bow_processors.yelp2014_processor import Yelp2014Processor
from models.lr.args import get_args
from models.lr.model import LogisticRegression

# String templates for logging results
LOG_HEADER = 'Split  Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss'
LOG_TEMPLATE = ' '.join('{:>5s},{:>9.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))


def evaluate_split(model, vectorizer, processor, args, save_file, split='dev'):
    evaluator = BagOfWordsEvaluator(model, vectorizer, processor, args, split)
    scores, score_names = evaluator.get_scores(silent=True)
    p_micro, r_micro, f1_micro, accuracy, avg_loss = scores[:5]
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy,
                              p_micro, r_micro, f1_micro,
                              avg_loss))

    scores_dict = dict(zip(score_names, scores))
    with open(save_file, 'w') as f:
        f.write(json.dumps(scores_dict))


def run_main(args):
    print('Args: ', args)
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print('Number of GPUs:', n_gpu)
    print('Device:', str(device).upper())

    metrics_dev_json = args.metrics_json + '_dev'
    metrics_test_json = args.metrics_json+'_test'

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    dataset_map = {
        'Reuters': ReutersProcessor,
        'CongressionalHearing': CongressionalHearingProcessor,
        'AAPD': AAPDProcessor,
        'IMDB': IMDBProcessor,
        'Yelp2014': Yelp2014Processor
    }

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    args.device = device
    args.n_gpu = n_gpu
    args.num_labels = dataset_map[args.dataset].NUM_CLASSES
    args.is_multilabel = dataset_map[args.dataset].IS_MULTILABEL
    args.vocab_size = min(args.max_vocab_size, dataset_map[args.dataset].VOCAB_SIZE)

    train_examples = None
    processor = dataset_map[args.dataset](args)
    vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"),
                                 # max_features=args.max_vocab_size,
                                 tokenizer=word_tokenize)

    if not args.trained_model:
        train_examples = processor.get_train_examples(args.data_dir)
        save_path = os.path.join(args.save_path, processor.NAME)
        os.makedirs(save_path, exist_ok=True)

    model = LogisticRegression(args)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)
    trainer = BagOfWordsTrainer(model, vectorizer, optimizer, processor, args)

    if not args.trained_model:
        trainer.train()
        model = torch.load(trainer.snapshot_path)
    else:
        model = torch.load(args.trained_model, map_location=lambda storage, location: storage)
        model = model.to(device)

    if args.evaluate_dev:
        evaluate_split(model, vectorizer, processor, args, metrics_dev_json, split='dev')
    if args.evaluate_test:
        evaluate_split(model, vectorizer, processor, args, metrics_test_json, split='test')


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    run_main(args)