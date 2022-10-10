

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from src.loss.focal_loss import FocalCELoss
from this import d
from torch import nn
from transformers.models.bert.modeling_bert import (BertModel,
                                                    BertPreTrainedModel)


class ModelingCtcS2eBert(BertPreTrainedModel):
    "多字和少字, 使用Electra"

    def __init__(self,
                 config):
        super(ModelingCtcS2eBert, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)  # v1.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.d_cls = nn.Linear(config.hidden_size, 2)
        self.c_cls = nn.Linear(config.hidden_size, config.correct_vocab_size)

    @staticmethod
    def build_dummpy_inputs():
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(8, 56)))
        inputs['attention_mask'] = torch.ones(size=(8, 56)).long()
        inputs['token_type_ids'] = torch.zeros(size=(8, 56)).long()
        return inputs
    
    def _init_criterion(self):
        self._d_criterion = FocalCELoss()
        self._c_criterion = FocalCELoss()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        d_labels = None,
        c_labels = None,

    ):

        encoder_outputs = self.bert(
            input_ids, attention_mask, token_type_ids)[0]
        d_logits = self.d_cls(encoder_outputs) 
        c_logits = self.c_cls(encoder_outputs)
        d_tag_num = d_logits.shape[-1]
        c_tag_num = c_logits.shape[-1]
        loss = None
        
        if d_labels is not None:
            d_loss = self._d_criterion(d_logits.view(-1, d_tag_num), d_labels.view(-1))
            c_loss = self._c_criterion(c_logits.view(-1, c_tag_num), c_labels.view(-1))
            loss = d_loss + c_loss
            
            
        return c_logits, d_logits, loss


if __name__ == "__main__":

    pretrained_dir = 'model/gector/pretrain_all_new_2022Y08M23D13H/normal_model/epoch2,step2183,testepochf1_0.001,devepochf1_0.0021'
    model = ModelingCtcS2eBert.from_pretrained(pretrained_dir)
    inputs = ModelingCtcS2eBert.build_dummpy_inputs()
    a = model(**inputs)
    print('1')
