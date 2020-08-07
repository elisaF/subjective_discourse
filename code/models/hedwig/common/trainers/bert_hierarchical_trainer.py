import datetime
import os
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
from tqdm import trange

from common.constants import *
from common.evaluators.bert_hierarchical_evaluator import BertHierarchicalEvaluator
from datasets.bert_processors.abstract_processor import convert_examples_to_features
from datasets.bert_processors.abstract_processor import convert_examples_to_hierarchical_features
from utils.preprocessing import pad_input_matrix, get_coarse_labels, get_fine_mask


class BertHierarchicalTrainer(object):
    def __init__(self, model, optimizer, processor, scheduler, tokenizer, args):
        self.args = args
        self.model = model
        self.model_fine = model
        self.optimizer = optimizer
        self.processor = processor
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.train_examples = self.processor.get_train_examples(args.data_dir)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.snapshot_path = os.path.join(self.args.save_path, self.processor.NAME, '%s.pt' % timestamp)

        self.num_train_optimization_steps = int(
            len(self.train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs

        self.log_header = 'Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss'
        self.log_template = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))

        self.iterations, self.nb_tr_steps, self.tr_loss_coarse, self.tr_loss_fine = 0, 0, 0, 0
        self.best_dev_f1, self.unimproved_iters = 0, 0
        self.early_stop = False

        self.initial_tr_loss_fine = float("inf")
        self.minimum_loss_percent_decrease = 0.4
        self.patience_training = 15
        self.training_converged = True

    def train_epoch(self, train_dataloader):
        self.tr_loss_coarse, self.tr_loss_fine = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.model.train()
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits_coarse, logits_fine = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)  # batch-size, num_classes

            # get coarse labels from the fine labels
            label_ids_coarse = get_coarse_labels(label_ids, self.args.num_coarse_labels,
                                                 self.args.parent_to_child_index_map, 
                                                 self.args.device)
            
            # calculate mask to ignore invalid
            # fine labels based on gold coarse labels
            mask_fine = get_fine_mask(label_ids_coarse, self.args.parent_to_child_index_map)

            if self.args.loss == 'cross-entropy':
                if self.args.pos_weights_coarse:
                    pos_weights_coarse = [float(w) for w in self.args.pos_weights_coarse.split(',')]
                    pos_weight_coarse = torch.FloatTensor(pos_weights_coarse)
                else:
                    pos_weight_coarse = torch.ones([self.args.num_coarse_labels])
                if self.args.pos_weights:
                    pos_weights = [float(w) for w in self.args.pos_weights.split(',')]
                    pos_weights = torch.FloatTensor(pos_weights)
                else:
                    pos_weights = torch.ones([self.args.num_labels])

                criterion_coarse = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_coarse)
                criterion_coarse = criterion_coarse.to(self.args.device)
                loss_coarse = criterion_coarse(logits_coarse, label_ids_coarse.float())

                criterion_fine = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
                criterion_fine = criterion_fine.to(self.args.device)
                logits_fine[~mask_fine] = -10000  # instead of -inf so loss is not nan
                loss_fine = criterion_fine(logits_fine, label_ids.float())

            loss_total = loss_coarse + loss_fine
            loss_total.backward()
            self.tr_loss_coarse += loss_coarse.item()
            self.tr_loss_fine += loss_fine.item()
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.iterations += 1

    def train(self):
        if self.args.is_hierarchical:
            train_features = convert_examples_to_hierarchical_features(
                self.train_examples, self.args.max_seq_length, self.tokenizer)
        else:
            train_features = convert_examples_to_features(
                self.train_examples, self.args.max_seq_length, self.tokenizer, use_guid=True)

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
        label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids, label_ids)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.batch_size)

        print('Begin training: ', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        start_time = time.monotonic()
        for epoch in trange(int(self.args.epochs), desc="Epoch"):
            self.train_epoch(train_dataloader)
            print('COARSE Train loss: ', self.tr_loss_coarse)
            print('FINE Train loss: ', self.tr_loss_fine)
            if epoch == 0:
                self.initial_tr_loss_fine = self.tr_loss_fine
            if self.args.evaluate_dev:
                dev_evaluator = BertHierarchicalEvaluator(self.model, self.processor,
                                                          self.tokenizer, self.args, split='dev')
                scores_fine, scores_coarse = dev_evaluator.get_scores(silent=True)
                dev_precision_fine, dev_recall_fine, dev_f1_fine, dev_acc_fine, dev_loss_fine = scores_fine[0][:5]
                dev_precision_coarse, dev_recall_coarse, dev_f1_coarse, dev_acc_coarse, dev_loss_coarse = scores_coarse[0][:5]

                # Print validation results
                tqdm.write('COARSE: '+self.log_header)
                tqdm.write(self.log_template.format(epoch + 1, self.iterations, epoch + 1, self.args.epochs,
                                                    dev_acc_coarse, dev_precision_coarse, dev_recall_coarse,
                                                    dev_f1_coarse, dev_loss_coarse))
                tqdm.write('FINE: ' + self.log_header)
                tqdm.write(self.log_template.format(epoch + 1, self.iterations, epoch + 1, self.args.epochs,
                                                    dev_acc_fine, dev_precision_fine, dev_recall_fine,
                                                    dev_f1_fine, dev_loss_fine))

                # Update validation results
                if dev_f1_fine > self.best_dev_f1:
                    self.unimproved_iters = 0
                    self.best_dev_f1 = dev_f1_fine
                    torch.save(self.model, self.snapshot_path)

                else:
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        tqdm.write("Early Stopping. Epoch: {}, Best Dev {}: {}".format(epoch, self.args.eval_metric, self.best_dev_f1))
                        break
            if self.args.evaluate_test:
                # when evaluating on test, we can't use dev
                # so check train loss is converging
                if epoch == self.patience_training:
                    loss_percent = (self.initial_tr_loss_fine-self.tr_loss_fine)/self.initial_tr_loss_fine
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
