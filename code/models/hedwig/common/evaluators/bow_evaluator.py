import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from datasets.bow_processors.abstract_processor import StreamingSparseDataset

# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')


class BagOfWordsEvaluator(object):
    def __init__(self, model, vectorizer, processor, args, split='dev'):
        self.args = args
        self.model = model
        self.processor = processor
        self.vectorizer = vectorizer

        if split == 'test':
            eval_examples = self.processor.get_test_examples(args.data_dir)
        else:
            eval_examples = self.processor.get_dev_examples(args.data_dir)

        self.eval_features = vectorizer.transform([x.text for x in eval_examples])
        self.eval_labels = [[float(x) for x in document.label] for document in eval_examples]

    def get_scores(self, silent=False):
        self.model.eval()
        eval_data = StreamingSparseDataset(self.eval_features, self.eval_labels)
        eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=self.args.batch_size)

        total_loss = 0
        nb_eval_steps = 0
        target_labels = list()
        predicted_labels = list()

        for features, labels in tqdm(eval_dataloader, desc="Evaluating", disable=silent):
            features = features.to(self.args.device)
            labels = labels.to(self.args.device)

            with torch.no_grad():
                logits = self.model(features)

            if self.args.n_gpu > 1:
                logits = logits.view(labels.size())

            if self.args.is_multilabel:
                predicted_labels.extend(torch.sigmoid(logits).round().long().cpu().detach().numpy())
                target_labels.extend(labels.cpu().detach().numpy())
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='sum')
            else:
                predicted_labels.extend(torch.argmax(logits, dim=1).cpu().detach().numpy())
                target_labels.extend(torch.argmax(labels, dim=1).cpu().detach().numpy())
                loss = F.cross_entropy(logits, torch.argmax(labels, dim=1))

            if self.args.n_gpu > 1:
                loss = loss.mean()

            total_loss += loss.item()
            nb_eval_steps += 1

        predicted_label_sets = [predicted_label.tolist() for predicted_label in predicted_labels]
        target_label_sets = [target_label.tolist() for target_label in target_labels]

        subset_accuracy = np.sum(np.equal(target_labels, predicted_labels).all(axis=1)) / len(target_labels)
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

        precision_class, recall_class, f1_class, support_class = metrics.precision_recall_fscore_support(target_labels, predicted_labels)

        avg_loss = total_loss / nb_eval_steps

        return [precision_macro, recall_macro, f1_macro,
                accuracy,
                avg_loss,
                hamming_loss,
                precision_micro, recall_micro, f1_micro,
                precision_class.tolist(), recall_class.tolist(), f1_class.tolist(), support_class.tolist(),
                cm.tolist(), target_label_sets, predicted_label_sets], \
               ['precision_macro', 'recall_macro', 'f1_macro',
                'accuracy',
                'avg_loss',
                'hamming_loss',
                'precision_micro', 'recall_micro', 'f1_micro',
                'precision_class', 'recall_class', 'f1_class', 'support_class',
                'confusion_matrix', 'target_label_sets', 'predicted_label_sets']
