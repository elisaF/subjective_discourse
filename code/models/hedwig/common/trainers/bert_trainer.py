import datetime
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
from tqdm import trange

from common.constants import *
from common.evaluators.bert_evaluator import BertEvaluator, evaluate_for_regression, evaluate_with_metric
from datasets.bert_processors.abstract_processor import convert_examples_to_features
from datasets.bert_processors.abstract_processor import convert_examples_to_hierarchical_features
from utils.optimization import warmup_linear
from utils.preprocessing import pad_input_matrix


class BertTrainer(object):
    def __init__(self, model, optimizer, processor, scheduler, tokenizer, args):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.processor = processor
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.train_examples = self.processor.get_train_examples(args.data_dir)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.snapshot_path = os.path.join(self.args.save_path, self.processor.NAME, '%s.pt' % timestamp)

        self.num_train_optimization_steps = int(
            len(self.train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs

        self.log_header_classification = 'Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss'
        self.log_template_classification = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))

        self.log_header_regression = 'Epoch Iteration Progress   Dev/RMSE   Dev/Kend.   Dev/Pears. Dev/Spear. Dev/Pears._Spear. Dev/Loss'
        self.log_template_regression = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f},{:8.4f},{:8.4f},{:8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))

        self.iterations, self.nb_tr_steps, self.tr_loss = 0, 0, 0
        self.unimproved_iters = 0
        if self.args.eval_metric == METRIC_RMSE:
            self.best_dev_metric = float('inf')
        else:
            self.best_dev_metric = 0
        self.early_stop = False

        self.initial_tr_loss = float("inf")
        self.minimum_loss_percent_decrease = 0.4
        self.patience_training = 10
        self.training_converged = True

    def train_epoch(self, train_dataloader):
        self.tr_loss = 0
        predicted_labels, target_labels = list(), list()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.model.train()
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]

            if self.args.is_multilabel:
                predicted_labels.extend(F.sigmoid(logits).round().long().cpu().detach().numpy())
                target_labels.extend(label_ids.cpu().detach().numpy())
                if self.args.loss == 'cross-entropy':
                    if self.args.pos_weights:
                        pos_weights = [float(w) for w in self.args.pos_weights.split(',')]
                        pos_weight = torch.FloatTensor(pos_weights)
                    else:
                        pos_weight = torch.ones([self.args.num_labels])
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    criterion = criterion.to(self.args.device)
                    loss = criterion(logits, label_ids.float())
                elif self.args.loss == 'mse':
                    criterion = torch.nn.MSELoss()
                    criterion = criterion.to(self.args.device)
                    m = torch.nn.Sigmoid()
                    m.to(self.args.device)
                    loss = criterion(m(logits), label_ids.float())
            else:
                if self.args.num_labels > 2:
                    predicted_labels.extend(torch.argmax(logits, dim=1).cpu().detach().numpy())
                    target_labels.extend(label_ids.cpu().detach().numpy())
                    loss = F.cross_entropy(logits, torch.argmax(label_ids, dim=1))
                else:
                    if self.args.is_regression:
                        predicted_labels.extend(logits.view(-1).cpu().detach().numpy())
                        target_labels.extend(label_ids.view(-1).cpu().detach().numpy())
                        criterion = torch.nn.MSELoss()
                        loss = criterion(logits.view(-1), label_ids.view(-1))
                    else:
                        criterion = torch.nn.CrossEntropyLoss()
                        loss = criterion(logits.view(-1, self.args.num_labels), label_ids.view(-1))

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.args.fp16:
                self.optimizer.backward(loss)
            else:
                loss.backward()
            self.tr_loss += loss.item()
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if self.args.fp16:
                    lr_this_step = self.args.learning_rate * warmup_linear(self.iterations / self.num_train_optimization_steps, self.args.warmup_proportion)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.iterations += 1
        if self.args.evaluate_train:
            rmse, kendall, pearson, spearman, pearson_spearman = evaluate_for_regression(target_labels, predicted_labels)
            print('\n' + LOG_HEADER_REG)
            print(LOG_TEMPLATE_REG.format('TRAIN', rmse, kendall, pearson, spearman, pearson_spearman, self.tr_loss))

    def train(self):
        if self.args.is_hierarchical:
            train_features = convert_examples_to_hierarchical_features(
                self.train_examples, self.args.max_seq_length, self.tokenizer)
        else:
            train_features = convert_examples_to_features(
                self.train_examples, self.args.max_seq_length, self.tokenizer, use_guid=True, is_regression=self.args.is_regression)

        unpadded_input_ids = [f.input_ids for f in train_features]
        unpadded_input_mask = [f.input_mask for f in train_features]
        unpadded_segment_ids = [f.segment_ids for f in train_features]

        if self.args.is_hierarchical:
            pad_input_matrix(unpadded_input_ids, self.args.max_doc_length)
            pad_input_matrix(unpadded_input_mask, self.args.max_doc_length)
            pad_input_matrix(unpadded_segment_ids, self.args.max_doc_length)

        print("Number of examples: ", len(self.train_examples))
        print("Batch size:", self.args.batch_size)
        print("Num of steps:", self.num_train_optimization_steps)

        padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
        padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
        padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)

        if self.args.is_regression:
            label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
        else:
            label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids, label_ids)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.batch_size)

        print('Begin training: ', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        start_time = time.monotonic()
        for epoch in trange(int(self.args.epochs), desc="Epoch"):
            self.train_epoch(train_dataloader)
            print('Train loss: ', self.tr_loss)
            if epoch == 0:
                self.initial_tr_loss = self.tr_loss
            if self.args.evaluate_dev:
                dev_evaluator = BertEvaluator(self.model, self.processor, self.tokenizer, self.args, split='dev')
                dev_scores, dev_score_names = dev_evaluator.get_scores()
                dev_metric = dev_scores[dev_score_names.index(self.args.eval_metric)]
                if self.args.is_regression:
                    dev_rmse, dev_kendall, dev_pearson, dev_spearman, dev_pearson_spearman, dev_loss = dev_scores[:6]

                    # Print validation results
                    tqdm.write(self.log_header_regression)
                    tqdm.write(self.log_template_regression.
                               format(epoch + 1, self.iterations, epoch + 1, self.args.epochs,
                                      dev_rmse, dev_kendall, dev_pearson, dev_spearman, dev_pearson_spearman, dev_loss))

                else:
                    dev_precision, dev_recall, dev_f1, dev_acc, dev_loss = dev_scores[:5]

                    # Print validation results
                    tqdm.write(self.log_header_classification)
                    tqdm.write(self.log_template_classification.
                               format(epoch + 1, self.iterations, epoch + 1, self.args.epochs,
                                      dev_acc, dev_precision, dev_recall, dev_f1, dev_loss))

                # Update validation results
                dev_improved = self.check_dev_improved(dev_metric)
                if dev_improved:
                    self.unimproved_iters = 0
                    self.best_dev_metric = dev_metric
                    torch.save(self.model, self.snapshot_path)

                else:
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        tqdm.write("Early Stopping. Epoch: {}, Best Dev {}: {}".format(epoch, self.args.eval_metric, self.best_dev_metric))
                        break
            if self.args.evaluate_test:
                if epoch == self.patience_training:
                    loss_percent = (self.initial_tr_loss-self.tr_loss)/self.initial_tr_loss
                    if loss_percent <= self.minimum_loss_percent_decrease:
                        self.training_converged = False
                        tqdm.write("Training failed to converge. Epoch: {}, Loss percent: {}"
                                   .format(epoch, loss_percent))
                        break
        end_time = time.monotonic()

        # save model at end of training
        # when evaluating on test
        if self.args.evaluate_test:
            torch.save(self.model, self.snapshot_path)
        print('End training: ', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print('Time elapsed: ', end_time-start_time)

    def check_dev_improved(self, dev_metric):
        # NOTE! This assumes correlations are always positive!
        if self.args.eval_metric == METRIC_RMSE:
            return dev_metric < self.best_dev_metric
        else:
            return dev_metric > self.best_dev_metric
