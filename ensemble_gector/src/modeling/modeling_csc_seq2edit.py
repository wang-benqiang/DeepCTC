#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from src.loss.focal_loss import FocalCELoss
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
from torch import nn



class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.correct_vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.correct_vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class ModelingCscSeq2Edit(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        if self.config.add_detect_task:
            self.tag_detect_projection_layer = torch.nn.Linear(
                config.hidden_size, config.detect_vocab_size) if config.add_detect_task else None
        self.tag_label_projection_layer = BertLMPredictionHead(config)
        self.init_weights()

          
    
    
    def _init_criterion(self,):
        
        self._detect_criterion = FocalCELoss(loss_labels_weights=[1, 3],
                                             gamma=2,
                                             ignore_index=-100)
        
        # loss_labels_weights = [3]*self.config.correct_vocab_size
        # loss_labels_weights[1] = 1
        self._correct_criterion = FocalCELoss(loss_labels_weights=None,
                                              gamma=2,
                                              ignore_index=-100)  

    @staticmethod
    def build_dummpy_inputs():
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(8, 56)))
        inputs['attention_mask'] = torch.ones(size=(8, 56)).long()
        inputs['token_type_ids'] = torch.zeros(size=(8, 56)).long()
        return inputs


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        detect_labels=None,
        correct_labels=None
    ):

        hidden_states = self.bert(input_ids, attention_mask, token_type_ids)[0]
  

        
        sequence_output = self.tag_label_projection_layer(hidden_states)

        loss = None
        
        if correct_labels is not None:
            loss = self._correct_criterion(
                sequence_output.view(-1, self.config.correct_vocab_size), correct_labels.view(-1))
       
        detect_outputs = self.tag_detect_projection_layer(hidden_states)
        
        if detect_labels is not None:
            loss += self._detect_criterion(
                detect_outputs.view(-1, self.config.detect_vocab_size), detect_labels.view(-1)) 
            
        
        return sequence_output, detect_outputs, loss


if __name__ == '__main__':
    pretrained_dir = 'pretrained_model/extend_macbert_base'
    # pretrained_dir = 'model/extend_electra_base'
    model = ModelingCscSeq2Edit.from_pretrained(pretrained_dir)
    inputs = ModelingCscSeq2Edit.build_dummpy_inputs()
    r = model(**inputs)
    print(1)
