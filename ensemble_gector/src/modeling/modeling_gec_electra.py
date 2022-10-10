

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.cuda.amp import autocast
from transformers.models.electra import ElectraModel, ElectraPreTrainedModel


class ModelingGecElectra(ElectraPreTrainedModel):
    "多字和少字, 使用Electra"

    def __init__(self,
                 config):
        super(ModelingGecElectra, self).__init__(config)
        self.config = config
        self.electra = ElectraModel(config)  # v1.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.add_detect_task:
            self.d_cls = nn.Linear(
                config.hidden_size * 2, 2)
        self.c_cls = nn.Linear(
            config.hidden_size * 2,
            config.vocab_size)

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
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.electra.get_extended_attention_mask(
            attention_mask, input_shape, device)
        head_mask = self.electra.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.electra.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if hasattr(self.electra, "embeddings_project"):
            embedding_output = self.electra.embeddings_project(
                embedding_output)

        encoder_outputs = self.electra.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        
        encoder_outputs = torch.cat((embedding_output, encoder_outputs),
                                    dim=-1)
        encoder_outputs = self.dropout(encoder_outputs)

        d_logits = self.d_cls(
            encoder_outputs) if self.config.add_detect_task else None
        c_logits = self.c_cls(encoder_outputs)

        return c_logits, d_logits


if __name__ == "__main__":

    pretrained_dir = 'model/extend_electra_small_gec'
    model = ModelingGecElectra.from_pretrained(pretrained_dir)
    inputs = ModelingGecElectra.build_dummpy_inputs()
    a, b = model(**inputs)
    print('1')
