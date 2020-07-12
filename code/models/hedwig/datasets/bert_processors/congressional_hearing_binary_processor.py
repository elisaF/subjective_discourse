import os

from datasets.bert_processors.abstract_processor import BertProcessor, InputExample


class CongressionalHearingBinaryProcessor(BertProcessor):
    NUM_CLASSES = 2
    IS_MULTILABEL = False

    def __init__(self, config):
        super().__init__()
        self.NAME = os.path.join('CongressionalHearingBinary', config.binary_label)

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
            guid = line[0]
            text_a = line[2]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples