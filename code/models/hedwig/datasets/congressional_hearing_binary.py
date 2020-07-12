import json
import os
import re
import sys
import csv

import numpy as np
import torch
from torchtext.data import NestedField, Field, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

from datasets.reuters import clean_string, split_sents, process_labels, generate_ngrams


def char_quantize(string, max_length=1000):
    identity = np.identity(len(CongressionalHearingCharQuantized.ALPHABET))
    quantized_string = np.array([identity[CongressionalHearingCharQuantized.ALPHABET[char]] for char in list(string.lower()) if char in CongressionalHearingCharQuantized.ALPHABET], dtype=np.float32)
    if len(quantized_string) > max_length:
        return quantized_string[:max_length]
    else:
        return np.concatenate((quantized_string, np.zeros((max_length - len(quantized_string), len(CongressionalHearingCharQuantized.ALPHABET)), dtype=np.float32)))


def process_docids(string):
    """
    Returns the docid as an integer
    :param string:
    :return:
    """
    try:
        docid = int(string)
    except ValueError:
        # print("Error converting docid to integer:", string)
        docid = 0
    return docid


class CongressionalHearingBinary(TabularDataset):
    NAME = 'CongressionalHearingBinary'
    NUM_CLASSES = 2
    IS_MULTILABEL = False

    TEXT_FIELD = Field(batch_first=True, tokenize=clean_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)
    DOCID_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_docids)
    
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train=os.path.join(NAME, 'train.tsv'),
               validation=os.path.join(NAME, 'dev.tsv'),
               test=os.path.join(NAME, 'test.tsv'), **kwargs):
        return super(CongressionalHearing, cls).splits(
            path, train=train, validation=validation, test=test,
            format='tsv', fields=[('docid', cls.DOCID_FIELD), ('label', cls.LABEL_FIELD), ('text', cls.TEXT_FIELD)]
        )

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None,
              unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)

    @classmethod
    def splits_dev(cls, path, train=os.path.join(NAME, 'train.tsv'),
                   validation=os.path.join(NAME, 'dev.tsv'), **kwargs):
        return super(CongressionalHearing, cls).splits(
            path, train=train, validation=validation, format='tsv', fields=[('docid', cls.DOCID_FIELD), ('label', cls.LABEL_FIELD), ('text', cls.TEXT_FIELD)]
        )

    @classmethod
    def iters_dev(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None,
              unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val = cls.splits_dev(path)
        cls.TEXT_FIELD.build_vocab(train, val, vectors=vectors)
        return BucketIterator.splits((train, val), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)

    @classmethod
    def splits_test(cls, path, train=os.path.join(NAME, 'train.tsv'),
                   test=os.path.join(NAME, 'test.tsv'), **kwargs):
        return super(CongressionalHearing, cls).splits(
            path, train=train, test=test, format='tsv', fields=[('docid', cls.DOCID_FIELD), ('label', cls.LABEL_FIELD), ('text', cls.TEXT_FIELD)]
        )

    @classmethod
    def iters_test(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None,
              unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, test = cls.splits_test(path)
        cls.TEXT_FIELD.build_vocab(train, test, vectors=vectors)
        return BucketIterator.splits((train, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)


class CongressionalHearingBOW(CongressionalHearing):
    TEXT_FIELD = Field(batch_first=True, tokenize=clean_string, preprocessing=generate_ngrams, include_lengths=True)


class CongressionalHearingHierarchical(CongressionalHearing):
    NESTING_FIELD = Field(batch_first=True, tokenize=clean_string)
    TEXT_FIELD = NestedField(NESTING_FIELD, tokenize=split_sents)


class CongressionalHearingCharQuantized(CongressionalHearing):
    ALPHABET = dict(map(lambda t: (t[1], t[0]), enumerate(list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""))))
    TEXT_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=char_quantize)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None,
              unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param batch_size: batch size
        :param device: GPU device
        :return:
        """
        train, val, test = cls.splits(path)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle, device=device)
