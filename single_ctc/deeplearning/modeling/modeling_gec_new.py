

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.cuda.amp import autocast
from transformers.models.electra import ElectraModel, ElectraPreTrainedModel
from src.deeplearning.loss.focal_loss import FocalCELoss


class ModelingGEC(ElectraPreTrainedModel):
    "合并多字少字模型, 使用Electra"

    def __init__(self,
                 config):
        super(ModelingGEC, self).__init__(config)
        self.config = config
        self.electra = ElectraModel(config)  # v1.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.d_cls = nn.Linear(
            config.hidden_size, 2)
        self.c_cls = nn.Linear(
            config.hidden_size,
            config.vocab_size)
        self._detect_criterion = FocalCELoss(loss_labels_weights=[1, 3],
                                             gamma=2,
                                             ignore_index=-100)
        self._correct_criterion = FocalCELoss(gamma=2, ignore_index=-100)
    @staticmethod
    def build_dummpy_inputs():
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(8, 56)))
        inputs['attention_mask'] = torch.ones(size=(8, 56)).long()
        inputs['token_type_ids'] = torch.zeros(size=(8, 56)).long()
        return inputs

    @autocast()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        detect_labels=None,
        correct_labels=None,
  
    ):


        encoder_outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        
        
        encoder_outputs = self.dropout(encoder_outputs)

        detect_outputs = self.d_cls(encoder_outputs) 
        sequence_output = self.c_cls(encoder_outputs)
        
        loss = None
        if detect_labels is not None and correct_labels is not None:

            loss = 0.2*self._detect_criterion(
                detect_outputs.view(-1, self.config.detect_vocab_size), detect_labels.view(-1)) + self._correct_criterion(
                sequence_output.view(-1, self.config.correct_vocab_size), correct_labels.view(-1))
        elif detect_labels is not None:
            loss = self._detect_criterion(
                detect_outputs.view(-1, self.config.detect_vocab_size), detect_labels.view(-1))
        elif correct_labels is not None:
            loss = self._correct_criterion(
                sequence_output.view(-1, self.config.correct_vocab_size), correct_labels.view(-1))

        return detect_outputs, sequence_output, loss


if __name__ == "__main__":

    pretrained_dir = 'model/extend_electra_small_gec'
    model = ModelingGEC.from_pretrained(pretrained_dir)
    inputs = ModelingGEC.build_dummpy_inputs()
    a, b = model(**inputs)
    print('1')
