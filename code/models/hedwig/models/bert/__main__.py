import random
import json

import numpy as np
import torch
from transformers import (
    AdamW, get_linear_schedule_with_warmup,
    BertForSequenceClassification, BertTokenizer,
    XLNetForSequenceClassification, XLNetTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    AlbertForSequenceClassification, AlbertTokenizer,
    ElectraForSequenceClassification, ElectraTokenizer
)

from common.constants import *
from common.evaluators.bert_evaluator import BertEvaluator
from common.trainers.bert_trainer import BertTrainer
from datasets.bert_processors.congressional_hearing_processor import CongressionalHearingProcessor
from datasets.bert_processors.congressional_hearing_binary_processor import CongressionalHearingBinaryProcessor
from models.bert.args import get_args


def evaluate_split(model, processor, tokenizer, args, save_file, split='dev'):
    evaluator = BertEvaluator(model, processor, tokenizer, args, split)
    scores, score_names = evaluator.get_scores(silent=True)
    if args.task == TASK_REGRESSION:
        rmse, kendall, pearson, spearman, pearson_spearman, avg_loss = scores[:6]
        print('\n' + LOG_HEADER_REG)
        print(LOG_TEMPLATE_REG.format(split.upper(), rmse, kendall, pearson, spearman, pearson_spearman, avg_loss))
    else:
        precision, recall, f1, accuracy, avg_loss = scores[:5]
        print('\n' + LOG_HEADER_CLASS)
        print(LOG_TEMPLATE_CLASS.format(split.upper(), accuracy, precision, recall, f1, avg_loss))

    scores_dict = dict(zip(score_names, scores))
    with open(save_file, 'w') as f:
        f.write(json.dumps(scores_dict))


def run_main(args):
    print('Args: ', args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    print('Device:', str(device).upper())
    print('Number of GPUs:', n_gpu)
    print('FP16:', args.fp16)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    metrics_dev_json = args.metrics_json + '_dev'
    metrics_test_json = args.metrics_json + '_test'

    dataset_map = {
        'CongressionalHearing': CongressionalHearingProcessor,
        'CongressionalHearingBinary': CongressionalHearingBinaryProcessor
    }

    model_map = {
        'bert': BertForSequenceClassification,
        'electra': ElectraForSequenceClassification,
        'xlnet': XLNetForSequenceClassification,
        'roberta': RobertaForSequenceClassification,
        'albert': AlbertForSequenceClassification
    }

    tokenizer_map = {
        'bert': BertTokenizer,
        'electra': ElectraTokenizer,
        'xlnet': XLNetTokenizer,
        'roberta': RobertaTokenizer,
        'albert': AlbertTokenizer
    }

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    if args.task == REGRESSION:
        args.num_labels = 1
        args.is_multilabel = False
        args.is_regression = True
    else:
        args.num_labels = dataset_map[args.dataset].NUM_CLASSES
        args.is_multilabel = dataset_map[args.dataset].IS_MULTILABEL
        args.is_regression = False
    args.is_hierarchical = False

    processor = dataset_map[args.dataset](args)

    if not args.trained_model:
        save_path = os.path.join(args.save_path, processor.NAME)
        os.makedirs(save_path, exist_ok=True)

    pretrained_vocab_path = args.model

    train_examples = None
    num_train_optimization_steps = None
    if not args.trained_model:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs

    pretrained_model_path = args.model

    tokenizer = tokenizer_map[args.model_family].from_pretrained(pretrained_vocab_path)
    model = model_map[args.model_family].from_pretrained(pretrained_model_path, num_labels=args.num_labels)

    # hacky fix for error in transformers code
    # that triggers error "Assertion srcIndex < srcSelectDimSize failed"
    # https://github.com/huggingface/transformers/issues/1538#issuecomment-570260748
    if args.model_family == 'roberta' and args.use_second_input:
        model.roberta.config.type_vocab_size = 2
        single_emb = model.roberta.embeddings.token_type_embeddings
        model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
        model.roberta.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

    if args.fp16:
        model.half()
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install NVIDIA Apex for FP16 training")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.lr,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_train_optimization_steps,
                                                    num_warmup_steps=args.warmup_proportion * num_train_optimization_steps)

    trainer = BertTrainer(model, optimizer, processor, scheduler, tokenizer, args)

    if not args.trained_model:
        trainer.train()
        model = torch.load(trainer.snapshot_path)

    else:
        model = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=args.num_labels)
        model_ = torch.load(args.trained_model, map_location=lambda storage, loc: storage)
        state = {}
        for key in model_.state_dict().keys():
            new_key = key.replace("module.", "")
            state[new_key] = model_.state_dict()[key]
        model.load_state_dict(state)
        model = model.to(device)

    if trainer.training_converged:
        if args.evaluate_dev:
            evaluate_split(model, processor, tokenizer, args, metrics_dev_json, split='dev')
        if args.evaluate_test:
            evaluate_split(model, processor, tokenizer, args, metrics_test_json, split='test')

    return trainer.training_converged


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    run_main(args)
