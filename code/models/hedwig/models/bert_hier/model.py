import torch
from torch import nn, tanh
from transformers import BertModel, RobertaModel, XLNetModel
from transformers.modeling_utils import SequenceSummary


class BertHierarchical(nn.Module):

    def __init__(self, model_name, num_fine_labels, num_coarse_labels):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name, num_labels=num_fine_labels)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier_coarse = nn.Linear(self.bert.config.hidden_size, num_coarse_labels)
        self.classifier_fine = nn.Linear(self.bert.config.hidden_size, num_fine_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,):
        """
        a batch is a tensor of shape [batch_size, #file_in_commit, #line_in_file]
        and each element is a line, i.e., a bert_batch,
        which consists of input_ids, input_mask, segment_ids, label_ids
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits_coarse = self.classifier_coarse(pooled_output)
        logits_fine = self.classifier_fine(pooled_output)

        return logits_coarse, logits_fine


class RobertaHierarchical(nn.Module):

    def __init__(self, model_name, num_fine_labels, num_coarse_labels, use_second_input=False):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained(model_name, num_labels=num_fine_labels)
        self.dropout = nn.Dropout(self.roberta.config.hidden_dropout_prob)
        self.classifier = RobertaClassificationHeads(self.roberta.config, num_coarse_labels, num_fine_labels)

        # hacky fix for error in transformers code
        # that triggers error "Assertion srcIndex < srcSelectDimSize failed"
        # https://github.com/huggingface/transformers/issues/1538#issuecomment-570260748
        if use_second_input:
            self.roberta.config.type_vocab_size = 2
            single_emb = self.roberta.embeddings.token_type_embeddings
            self.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
            self.roberta.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        """
        a batch is a tensor of shape [batch_size, #file_in_commit, #line_in_file]
        and each element is a line, i.e., a bert_batch,
        which consists of input_ids, input_mask, segment_ids, label_ids
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits_coarse, logits_fine = self.classifier(sequence_output)

        return logits_coarse, logits_fine


class RobertaClassificationHeads(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_coarse_labels, num_fine_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_coarse = nn.Linear(config.hidden_size, num_coarse_labels)
        self.classifier_fine = nn.Linear(config.hidden_size, num_fine_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = tanh(x)
        output = self.dropout(x)
        logits_coarse = self.classifier_coarse(output)
        logits_fine = self.classifier_fine(output)
        return logits_coarse, logits_fine


class XLNetHierarchical(nn.Module):

    def __init__(self, model_name, num_fine_labels, num_coarse_labels):
        super().__init__()

        self.transformer = XLNetModel.from_pretrained(model_name, num_labels=num_fine_labels)
        self.sequence_summary = SequenceSummary(self.transformer.config)
        self.classifier_coarse = nn.Linear(self.transformer.config.d_model, num_coarse_labels)
        self.classifier_fine = nn.Linear(self.transformer.config.d_model, num_fine_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            use_cache=True,
            labels=None,
    ):
        """
        a batch is a tensor of shape [batch_size, #file_in_commit, #line_in_file]
        and each element is a line, i.e., a bert_batch,
        which consists of input_ids, input_mask, segment_ids, label_ids
        """

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        output = transformer_outputs[0]
        output = self.sequence_summary(output)

        logits_coarse = self.classifier_coarse(output)
        logits_fine = self.classifier_fine(output)

        return logits_coarse, logits_fine
