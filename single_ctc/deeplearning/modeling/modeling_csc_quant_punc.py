
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers.models.bert import BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.bert import BertPreTrainedModel
from torch.cuda.amp import autocast



class BertForCscQuantPunc(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()
        self.bert.pooler.dense.weight.requires_grad = False
        self.bert.pooler.dense.bias.requires_grad = False
    @autocast()
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output = self.bert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    return_dict=False)[0]
        logits_labels = self.cls(sequence_output)
        return logits_labels