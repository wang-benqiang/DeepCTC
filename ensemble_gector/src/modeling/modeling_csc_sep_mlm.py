#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from src.loss.focal_loss import FocalCELoss
from torch.cuda.amp import autocast


class ModelMlm(BertPreTrainedModel):
    "单纯使用Roberta预训练模型"
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    
    def __init__(
        self,
        config
    ):
        
        super(ModelMlm, self).__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()
        self._loss_func = FocalCELoss(gamma=2, ignore_index=-100)
        
        
    @autocast()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None
                ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]
        sequence_output = self.cls(sequence_output)
        loss = self._loss_func(sequence_output.view(-1, self.config.vocab_size), labels.view(-1)) if labels is not None else None
        return sequence_output, loss
