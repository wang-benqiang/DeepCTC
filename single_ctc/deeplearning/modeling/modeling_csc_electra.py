#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers.models.electra import ElectraModel, ElectraPreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from torch.cuda.amp import autocast
import torch

"electra small"

class ElectraForCsc(ElectraPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.electra = ElectraModel(config)
        if config.add_detect_task:
            self.d_cls = torch.nn.Linear(config.hidden_size, 2)
        self.c_cls = BertOnlyMLMHead(config)
        self.init_weights()
        
        
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
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.electra.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.electra.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.electra.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if hasattr(self.electra, "embeddings_project"):
            hidden_states = self.electra.embeddings_project(hidden_states)

        hidden_states = self.electra.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        
        sequence_output = hidden_states[0]
      
      
        
        detect_outputs = self.d_cls(
            sequence_output) if self.config.add_detect_task else None

        
        sequence_output = self.c_cls(sequence_output)

        return detect_outputs, sequence_output,
    


if __name__ == '__main__':
    pretrained_dir = 'pretrained_model/extend_macbert_base'
    # pretrained_dir = 'model/extend_electra_base'
    model  =  ElectraForCsc.from_pretrained(pretrained_dir)
    inputs = ElectraForCsc.build_dummpy_inputs()
    a, b = model(**inputs)
    print(1)