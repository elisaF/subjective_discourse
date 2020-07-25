import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.onnx

from common.evaluate import EvaluatorFactory
from common.train import TrainerFactory
from datasets.congressional_hearing import CongressionalHearingBOW
from models.fasttext.args import get_args
from models.fasttext.model import FastText


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]


def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, batch_size, device, is_multilabel, save_file):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device)
    if hasattr(saved_model_evaluator, 'is_multilabel'):
        saved_model_evaluator.is_multilabel = is_multilabel

    scores, score_names = saved_model_evaluator.get_scores()
    print('Evaluation metrics for', split_name)
    print(score_names)
    print(scores)

    scores_dict = dict(zip(score_names, scores))
    with open(save_file, 'w') as f:
        f.write(json.dumps(scores_dict))


def run_main(args):
    print('Args: ', args)

    metrics_dev_json = args.metrics_json + '_dev'
    metrics_test_json = args.metrics_json + '_test'

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not args.cuda:
        args.gpu = -1

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if torch.cuda.is_available() and args.cuda:
        print('Note: You are using GPU for training')
        torch.cuda.manual_seed(args.seed)
        args.device = torch.device('cuda:%d' % args.gpu)

    if torch.cuda.is_available() and not args.cuda:
        print('Warning: Using CPU for training')

    dataset_map = {
        'CongressionalHearing': CongressionalHearingBOW,
    }

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    else:
        dataset_class = dataset_map[args.dataset]
        if args.fold_num >= 0:
            dataset_name = os.path.join(dataset_class.NAME + 'Folds', 'fold' + str(args.fold_num))
        else:
            dataset_name = dataset_class.NAME
        if args.evaluate_dev:
            train_iter, dev_iter = dataset_map[args.dataset].iters_dev(args.data_dir, dataset_name, 
                                                                       args.word_vectors_file,
                                                                       args.word_vectors_dir,
                                                                       batch_size=args.batch_size, device=args.device,
                                                                       unk_init=UnknownWordVecCache.unk)
        if args.evaluate_test:
            train_iter, test_iter = dataset_map[args.dataset].iters_test(args.data_dir, dataset_name,
                                                                         args.word_vectors_file,
                                                                         args.word_vectors_dir,
                                                                         batch_size=args.batch_size, device=args.device,
                                                                         unk_init=UnknownWordVecCache.unk)

    config = deepcopy(args)
    config.dataset = train_iter.dataset
    config.target_class = train_iter.dataset.NUM_CLASSES
    config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)

    print('Dataset:', args.dataset)
    print('No. of target classes:', train_iter.dataset.NUM_CLASSES)
    print('No. of train instances', len(train_iter.dataset))
    if args.evaluate_dev:
        print('No. of dev instances', len(dev_iter.dataset))
    if args.evaluate_test:
        print('No. of test instances', len(test_iter.dataset))

    if args.resume_snapshot:
        if args.cuda:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.device))
        else:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
    else:
        model = FastText(config)
        model.to(device)

    if not args.trained_model:
        save_path = os.path.join(args.save_path, dataset_name)
        os.makedirs(save_path, exist_ok=True)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

    train_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, train_iter, args.batch_size, args.device)
    if args.evaluate_dev:
        dev_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, dev_iter, args.batch_size, args.device)
        if hasattr(dev_evaluator, 'is_multilabel'):
            dev_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
    if args.evaluate_test:
        test_evaluator = EvaluatorFactory.get_evaluator(dataset_map[args.dataset], model, None, test_iter, args.batch_size, args.device)
        if hasattr(test_evaluator, 'is_multilabel'):
            test_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
    if hasattr(train_evaluator, 'is_multilabel'):
        train_evaluator.is_multilabel = dataset_class.IS_MULTILABEL

    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_every,
        'patience': args.patience,
        'model_outfile': args.save_path,
        'is_multilabel': dataset_class.IS_MULTILABEL
    }
    if args.evaluate_dev:
        trainer = TrainerFactory.get_trainer_dev(args.dataset, model, None, train_iter, trainer_config, train_evaluator,
                                                 dev_evaluator, args)
    if args.evaluate_test:
        trainer = TrainerFactory.get_trainer_test(args.dataset, model, None, train_iter, trainer_config, train_evaluator,
                                             test_evaluator, args)

    if not args.trained_model:
        trainer.train(args.epochs)
    else:
        if args.cuda:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.device))
        else:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage)

    # Calculate dev and test metrics
    if hasattr(trainer, 'snapshot_path'):
        model = torch.load(trainer.snapshot_path)

    if args.evaluate_dev:
        evaluate_dataset('dev', dataset_map[args.dataset], model, None, dev_iter, args.batch_size,
                         is_multilabel=dataset_class.IS_MULTILABEL,
                         device=args.device, save_file=metrics_dev_json)
    if args.evaluate_test:
        evaluate_dataset('test', dataset_map[args.dataset], model, None, test_iter, args.batch_size,
                         is_multilabel=dataset_class.IS_MULTILABEL,
                         device=args.device, save_file=metrics_test_json)


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    run_main(args)

