#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from TorchCRF import CRF
from src import logger
from torch import nn
from transformers.models.electra import ElectraModel, ElectraPreTrainedModel, ElectraForTokenClassification
from transformers.activations import get_activation
from torch.cuda.amp import autocast
from src.deeplearning.loss.focal_loss import FocalCELoss

class WordOrderErrorModel(ElectraPreTrainedModel):
    """合并了detector和Corrector
     "检错的话,预测对应位置的append, keep, replace, delete"
     "纠错的话,直接预测对应位置的CTC label的 1.6W标签"
    """

    def __init__(self,
                 config):
        super(WordOrderErrorModel, self).__init__(config)
        self.config = config
        self.electra = ElectraModel(config) 
        self.classifier = nn.Linear(
            config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.append_crf:
            self.crf_layer = CRF(config.num_labels)
        self.criterion = FocalCELoss(ignore_index=-100)
        
        
    @autocast()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels = None
    ):

        sequence_output = self.electra(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )[0]
        
        
        sequence_output = self.dropout(sequence_output)

        sequence_output = self.classifier(sequence_output)

        if self.config.append_crf:
            attention_mask = attention_mask.bool()
            pred_labels = self.crf_layer.viterbi_decode(sequence_output, attention_mask)
            if labels is None:
                # prediction mode
                return pred_labels, sequence_output
            crf_loglikelihood = self.crf_layer.forward(
                sequence_output, labels, attention_mask)

            crf_loglikelihood = crf_loglikelihood.mean()
            # training mode
            return pred_labels, -crf_loglikelihood
        else:
            loss = self.criterion(sequence_output.view(-1, self.config.num_labels),
                                  labels.view(-1)
                                  )
        return sequence_output, loss



if __name__ == "__main__":
    from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
    pretrained_dir = 'model/extend_electra_base'
    tokenizer = CustomBertTokenizer.from_pretrained(pretrained_dir)
    
    inputs = tokenizer(['去玩儿','撒打'], return_tensors='pt')
    labels = torch.LongTensor([[1,2,0,1,1],[1,2,1,2,1]])
    model = WordOrderErrorModel.from_pretrained(pretrained_dir, )
    model.cuda()
    inputs['labels'] = labels
    inputs['input_ids'] = inputs['input_ids'].cuda()
    inputs['attention_mask'] = inputs['attention_mask'].cuda()
    inputs['token_type_ids'] = inputs['token_type_ids'].cuda()
    inputs['labels'] = inputs['labels'].cuda()
    r = model(**inputs)
    print('1')
