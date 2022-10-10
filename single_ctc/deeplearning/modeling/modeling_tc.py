#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.cuda.amp import autocast
from transformers.models.bert import BertModel, BertPreTrainedModel


# model for text classification

class ModelingBertForTC(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 20)

        self.init_weights()

    @staticmethod
    def build_dummpy_inputs():
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(8, 56)))
        inputs['attention_mask'] = torch.ones(size=(8, 56)).long()
        inputs['token_type_ids'] = torch.zeros(size=(8, 56)).long()
        inputs['labels'] = torch.zeros(size=(8, 3))
        return inputs

    def _init_criterion(self, class_weight=None, pos_weight=None):
        
        self.loss_fct = nn.BCEWithLogitsLoss(weight=class_weight, pos_weight=pos_weight)

    @autocast()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,

        )[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:

            loss = self.loss_fct(logits, labels)


        return logits, loss


if __name__ == '__main__':
    pretrained_dir = 'pretrained_model/chinese-bert-wwm'
    # pretrained_dir = 'model/extend_electra_base'
    model = ModelingBertForTC.from_pretrained(pretrained_dir)
    model._init_criterion()
    inputs = ModelingBertForTC.build_dummpy_inputs()
    r = model(**inputs)
    print(1)
