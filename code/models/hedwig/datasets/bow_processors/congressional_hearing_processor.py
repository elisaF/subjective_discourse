import os

from datasets.bow_processors.abstract_processor import BagOfWordsProcessor, InputExample


class CongressionalHearingProcessor(BagOfWordsProcessor):
    NUM_CLASSES = 6
    VOCAB_SIZE = 36308
    IS_MULTILABEL = True

    def __init__(self, config=None):
        super().__init__()
        if config and config.fold_num >= 0:
            self.NAME = os.path.join('CongressionalHearingFolds', 'fold'+str(config.fold_num))
        else:
            self.NAME = 'CongressionalHearing'

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.NAME, 'train.tsv')))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.NAME, 'dev.tsv')))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.NAME, 'test.tsv')))

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            examples.append(InputExample(guid=line[0], text=line[2], label=line[1]))
        return examples
