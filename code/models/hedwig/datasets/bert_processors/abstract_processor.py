import csv

import sys
import numpy as np
from nltk.tokenize import sent_tokenize


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, text_d=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.text_d = text_d
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputFeaturesWithId(InputFeatures):
    """A single set of features for relevance tasks."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid):
        super().__init__(input_ids, input_mask, segment_ids, label_id)
        self.guid = int(guid)


class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the train set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the dev set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the test set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_labels(self):
        """
        Gets a list of possible labels in the dataset
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


def convert_examples_to_features(examples, max_seq_length, tokenizer, print_examples=False, use_guid=False, is_regression=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        tokens_c = None
        tokens_d = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            if not example.text_c:
                orig_len_a = len(tokens_a)
                orig_len_b = len(tokens_b)
                removed_tokens_a, removed_tokens_b = _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
                if removed_tokens_a or removed_tokens_b:
                    print('Truncating example', example.guid, 'from ', orig_len_a + orig_len_b, 'to', (max_seq_length - 3))
                    print('Truncated text a: ', removed_tokens_a)
                    print('Truncated text b: ', removed_tokens_b)
            else:
                tokens_c = tokenizer.tokenize(example.text_c)
                # Modifies `tokens_a`, `tokens_b`, `tokens_c` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
                if not example.text_d:
                    orig_len_a = len(tokens_a)
                    orig_len_b = len(tokens_b)
                    orig_len_c = len(tokens_c)
                    removed_tokens_a, removed_tokens_b, removed_tokens_c = _truncate_seq_triplet(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
                    if removed_tokens_a or removed_tokens_b or removed_tokens_c:
                        print('Truncating example', example.guid, 'from ', orig_len_a + orig_len_b + orig_len_c, 'to', (max_seq_length - 4))
                        print('Truncated text a: ', removed_tokens_a)
                        print('Truncated text b: ', removed_tokens_b)
                        print('Truncated text c: ', removed_tokens_c)
                else:
                    tokens_d = tokenizer.tokenize(example.text_d)
                    # Modifies `tokens_a`, `tokens_b`, `tokens_c`, `tokens_d` in place so that the total
                    # length is less than the specified length.
                    # Account for [CLS], [SEP], [SEP], [SEP] with "- 5"
                    orig_len_a = len(tokens_a)
                    orig_len_b = len(tokens_b)
                    orig_len_c = len(tokens_c)
                    orig_len_d = len(tokens_d)
                    removed_tokens_a, removed_tokens_b, removed_tokens_c, removed_tokens_d = _truncate_seq_quadruplet(tokens_a, tokens_b,
                                                                                                 tokens_c, tokens_d,
                                                                                                 max_seq_length - 5)
                    if removed_tokens_a or removed_tokens_b or removed_tokens_c or removed_tokens_d:
                        print('Truncating example', example.guid, 'from ', orig_len_a + orig_len_b + orig_len_c + orig_len_d, 'to',
                              (max_seq_length - 5))
                        print('Truncated text a: ', removed_tokens_a)
                        print('Truncated text b: ', removed_tokens_b)
                        print('Truncated text c: ', removed_tokens_c)
                        print('Truncated text d: ', removed_tokens_d)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                print('Truncating example', example.guid, 'from ', len(tokens_a), 'to', (max_seq_length - 2))
                print('Truncated text is ', tokens_a[(max_seq_length - 2):])
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [tokenizer.sep_token]
            segment_ids += [1] * (len(tokens_b) + 1)

            if tokens_c:
                tokens += tokens_c + [tokenizer.sep_token]
                segment_ids += [0] * (len(tokens_c) + 1)

                if tokens_d:
                    tokens += tokens_d + [tokenizer.sep_token]
                    segment_ids += [0] * (len(tokens_d) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if is_regression:
            label_id = float(example.label)
        else:
            label_id = [float(x) for x in example.label]
            
        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s" % example.label)

        if use_guid:
            features.append(InputFeaturesWithId(input_ids=input_ids,
                                                input_mask=input_mask,
                                                segment_ids=segment_ids,
                                                label_id=label_id,
                                                guid=example.guid))
        else:
            features.append(InputFeatures(input_ids=input_ids,
                                          input_mask=input_mask,
                                          segment_ids=segment_ids,
                                          label_id=label_id))
    return features


def convert_examples_to_hierarchical_features(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = [tokenizer.tokenize(line) for line in sent_tokenize(example.text_a)]
        tokens_b = None

        if example.text_b:
            tokens_b = [tokenizer.tokenize(line) for line in sent_tokenize(example.text_b)]
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length
            # Account for [CLS], [SEP], [SEP]
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP]
            for i0 in range(len(tokens_a)):
                if len(tokens_a[i0]) > max_seq_length - 2:
                    tokens_a[i0] = tokens_a[i0][:(max_seq_length - 2)]

        tokens = [[tokenizer.cls_token] + line + [tokenizer.sep_token] for line in tokens_a]
        segment_ids = [[0] * len(line) for line in tokens]

        if tokens_b:
            tokens += tokens_b + [tokenizer.sep_token]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = list()
        for line in tokens:
            input_ids.append(tokenizer.convert_tokens_to_ids(line))

        # Input mask has 1 for real tokens and 0 for padding tokens
        input_mask = [[1] * len(line_ids) for line_ids in input_ids]

        # Zero-pad up to the sequence length.
        padding = [[0] * (max_seq_length - len(line_ids)) for line_ids in input_ids]
        for i0 in range(len(input_ids)):
            input_ids[i0] += padding[i0]
            input_mask[i0] += padding[i0]
            segment_ids[i0] += padding[i0]

        label_id = [float(x) for x in example.label]

        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s" % example.label)

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length
    :param tokens_a:
    :param tokens_b:
    :param max_length:
    :return:
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    removed_tokens_a = []
    removed_tokens_b = []
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            removed_tokens_a.append(tokens_a.pop())
        else:
            removed_tokens_b.append(tokens_b.pop())
    return removed_tokens_a, removed_tokens_b


def _truncate_seq_triplet(tokens_a, tokens_b, tokens_c, max_length):
    """
    Truncates a sequence triplet in place to the maximum length
    :param tokens_a:
    :param tokens_b:
    :param tokens_c:
    :param max_length:
    :return:
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    removed_tokens_a = []
    removed_tokens_b = []
    removed_tokens_c = []
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        longest_seq = np.array([len(tokens_a), len(tokens_b), len(tokens_c)]).argmax(axis=0)
        if longest_seq == 0:
            removed_tokens_a.append(tokens_a.pop())
        elif longest_seq == 1:
            removed_tokens_b.append(tokens_b.pop())
        else:
            removed_tokens_c.append(tokens_c.pop())
    return removed_tokens_a, removed_tokens_b, removed_tokens_c


def _truncate_seq_quadruplet(tokens_a, tokens_b, tokens_c, tokens_d, max_length):
    """
    Truncates a sequence triplet in place to the maximum length
    :param tokens_a:
    :param tokens_b:
    :param tokens_c:
    :param tokens_d:
    :param max_length:
    :return:
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    removed_tokens_a = []
    removed_tokens_b = []
    removed_tokens_c = []
    removed_tokens_d = []
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c) + len(tokens_d)
        if total_length <= max_length:
            break
        longest_seq = np.array([len(tokens_a), len(tokens_b), len(tokens_c), len(tokens_d)]).argmax(axis=0)
        if longest_seq == 0:
            removed_tokens_a.append(tokens_a.pop())
        elif longest_seq == 1:
            removed_tokens_b.append(tokens_b.pop())
        elif longest_seq == 2:
            removed_tokens_c.append(tokens_c.pop())
        else:
            removed_tokens_d.append(tokens_d.pop())
    return removed_tokens_a, removed_tokens_b, removed_tokens_c, removed_tokens_d

