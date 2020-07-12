import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from datasets.bert_processors.abstract_processor import convert_examples_to_features
from utils.preprocessing import pad_input_matrix, get_coarse_labels, get_fine_mask

# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')


class BertHierarchicalEvaluator(object):
    def __init__(self, model, processor, tokenizer, args, split='dev'):
        self.args = args
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

        if split == 'test':
            self.eval_examples = self.processor.get_test_examples(args.data_dir)
        else:
            self.eval_examples = self.processor.get_dev_examples(args.data_dir)

    def get_scores(self, silent=False):
        eval_features = convert_examples_to_features(self.eval_examples, self.args.max_seq_length,
                                                     self.tokenizer, use_guid=True)

        unpadded_input_ids = [f.input_ids for f in eval_features]
        unpadded_input_mask = [f.input_mask for f in eval_features]
        unpadded_segment_ids = [f.segment_ids for f in eval_features]

        padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
        padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
        padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
        label_ids_fine = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        doc_ids = torch.tensor([f.guid for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids, label_ids_fine, doc_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.batch_size)

        self.model.eval()

        total_loss_fine, total_loss_coarse = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predicted_labels_coarse, predicted_labels_fine = list(), list()
        target_labels_coarse, target_labels_fine = list(), list()
        target_doc_ids = list()

        for input_ids, input_mask, segment_ids, label_ids_fine, doc_ids in tqdm(eval_dataloader, desc="Evaluating",
                                                                                disable=silent):
            input_ids = input_ids.to(self.args.device)
            input_mask = input_mask.to(self.args.device)
            segment_ids = segment_ids.to(self.args.device)
            label_ids_fine = label_ids_fine.to(self.args.device)
            target_doc_ids.extend(doc_ids.tolist())

            with torch.no_grad():
                logits_coarse, logits_fine = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

            preds_coarse = F.sigmoid(logits_coarse).round().long().cpu().detach().numpy()
            predicted_labels_coarse.extend(preds_coarse)
            # get coarse labels from the fine labels
            label_ids_coarse = get_coarse_labels(label_ids_fine, self.args.num_coarse_labels,
                                                 self.args.parent_to_child_index_map, self.args.device)

            target_labels_coarse.extend(label_ids_coarse.cpu().detach().numpy())

            # mask fine predictions using coarse predictions
            preds_fine = F.sigmoid(logits_fine).round().long().cpu().detach().numpy()
            mask_fine = get_fine_mask(torch.Tensor(preds_coarse), self.args.parent_to_child_index_map)
            preds_fine[~mask_fine] = 0
            predicted_labels_fine.extend(preds_fine)

            target_labels_fine.extend(label_ids_fine.cpu().detach().numpy())

            if self.args.loss == 'cross-entropy':
                criterion = torch.nn.BCEWithLogitsLoss(size_average=False)
                loss_fine = criterion(logits_fine.cpu(), label_ids_fine.float().cpu())
                loss_coarse = criterion(logits_coarse.cpu(), label_ids_coarse.float().cpu())
            elif self.args.loss == 'mse':
                criterion = torch.nn.MSELoss(size_average=False)
                m = torch.nn.Sigmoid()
                loss_fine = criterion(m(logits_fine.cpu()), label_ids_fine.float().cpu())
                loss_coarse = criterion(m(logits_coarse.cpu()), label_ids_coarse.float().cpu())

            total_loss_fine += loss_fine.item()
            total_loss_coarse += loss_coarse.item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        metrics_fine = get_metrics(target_labels_fine, predicted_labels_fine,
                                   target_doc_ids, total_loss_fine, nb_eval_steps)
        metrics_coarse = get_metrics(target_labels_coarse, predicted_labels_coarse,
                                     target_doc_ids, total_loss_coarse, nb_eval_steps)

        metric_names = ['precision_macro', 'recall_macro', 'f1_macro',
                        'accuracy',
                        'avg_loss',
                        'hamming_loss',
                        'precision_micro', 'recall_micro', 'f1_micro',
                        'precision_class', 'recall_class', 'f1_class', 'support_class',
                        'confusion_matrix', 'id_gold_pred']

        metric_names_fine = [name + '_fine' for name in metric_names]
        metric_names_coarse = [name + '_coarse' for name in metric_names]
        return [metrics_fine, metric_names_fine], [metrics_coarse, metric_names_coarse]


def get_metrics(target_labels, predicted_labels, doc_ids, total_loss, n_steps):
    predicted_label_sets = [predicted_label.tolist() for predicted_label in predicted_labels]
    target_label_sets = [target_label.tolist() for target_label in target_labels]

    hamming_loss = metrics.hamming_loss(target_labels, predicted_labels)

    predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
    cm = metrics.multilabel_confusion_matrix(target_labels, predicted_labels)
    accuracy = metrics.accuracy_score(target_labels, predicted_labels)

    precision_micro = metrics.precision_score(target_labels, predicted_labels, average='micro')
    recall_micro = metrics.recall_score(target_labels, predicted_labels, average='micro')
    f1_micro = metrics.f1_score(target_labels, predicted_labels, average='micro')

    precision_macro = metrics.precision_score(target_labels, predicted_labels, average='macro')
    recall_macro = metrics.recall_score(target_labels, predicted_labels, average='macro')
    f1_macro = metrics.f1_score(target_labels, predicted_labels, average='macro')

    precision_class, recall_class, f1_class, support_class = metrics.precision_recall_fscore_support(target_labels,
                                                                                                     predicted_labels)

    avg_loss = total_loss / n_steps

    return [precision_macro, recall_macro, f1_macro,
            accuracy,
            avg_loss,
            hamming_loss,
            precision_micro, recall_micro, f1_micro,
            precision_class.tolist(), recall_class.tolist(), f1_class.tolist(), support_class.tolist(),
            cm.tolist(), list(zip(doc_ids, target_label_sets, predicted_label_sets))]
