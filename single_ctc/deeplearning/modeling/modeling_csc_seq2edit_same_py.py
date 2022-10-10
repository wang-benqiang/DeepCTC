#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from src.deeplearning.loss.focal_loss import FocalCELoss
from src.deeplearning.loss.smoothing_loss import LabelSmoothingLoss
from torch import nn
from torch.cuda.amp import autocast
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            config.hidden_size, config.correct_vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.correct_vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ModelingCscSeq2EditSamePy(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        if config.add_detect_task:

            self.tag_detect_projection_layer = torch.nn.Linear(
                config.hidden_size, config.detect_vocab_size) if config.add_detect_task else None

        if config.add_pinyin_task:
            self.tag_pinyin_projection_layer = torch.nn.Linear(
                config.hidden_size, config.pinyin_vocab_size) if config.add_detect_task else None

        self.tag_label_projection_layer = BertLMPredictionHead(config)
        self.init_weights()
    
    
    def _init_criterion(self,):
    
        self._detect_loss = FocalCELoss(loss_labels_weights=[1, 3],
                                            gamma=2,
                                            ignore_index=-100)

        self._smoothing_loss = LabelSmoothingLoss()
        self._py_smoothing_loss = LabelSmoothingLoss()

        

    @staticmethod
    def build_dummpy_inputs():
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(8, 56)))
        inputs['attention_mask'] = torch.ones(size=(8, 56)).long()
        inputs['token_type_ids'] = torch.zeros(size=(8, 56)).long()
        inputs['detect_labels'] = torch.zeros(size=(8, 56)).long()
        inputs['pinyin_labels'] = torch.zeros(size=(8, 56)).long()
        inputs['edit_labels'] = torch.zeros(size=(8, 56)).long()
        return inputs

    @autocast()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        detect_labels=None,
        pinyin_labels=None,
        edit_labels=None,
    ):

        hidden_states = self.bert(input_ids, attention_mask, token_type_ids)[0]
        c_loss, d_loss, py_loss = 0, 0, 0
        if self.config.add_detect_task and self.config.add_pinyin_task and edit_labels is not None:
            detect_outputs = self.tag_detect_projection_layer(hidden_states)
            pinyin_outputs = self.tag_pinyin_projection_layer(hidden_states)
            d_loss = self._detect_loss(
                detect_outputs.view(-1, self.config.detect_vocab_size), detect_labels.view(-1))
            py_loss = self._py_smoothing_loss(
                pinyin_outputs.view(-1, self.config.pinyin_vocab_size), pinyin_labels.view(-1))

        
        
        sequence_output = self.tag_label_projection_layer(hidden_states)



        if edit_labels is not None:
            c_loss = self._smoothing_loss(
                sequence_output.view(-1, self.config.correct_vocab_size), edit_labels.view(-1))

        detect_outputs = None
        
        
        total_loss = c_loss + d_loss + py_loss
        return sequence_output, detect_outputs, total_loss


if __name__ == '__main__':
    pretrained_dir = 'model/ctc_csc_no_punc_pretrain_2022Y06M18D18H/epoch3,ith_db0,step88600,testf1_63_94%,devf1_82_28%'
    # pretrained_dir = 'model/extend_electra_base'
    model = ModelingCscSeq2EditSamePy.from_pretrained(pretrained_dir)
    model._init_criterion()
    inputs = ModelingCscSeq2EditSamePy.build_dummpy_inputs()
    r = model(**inputs)
    print(1)
