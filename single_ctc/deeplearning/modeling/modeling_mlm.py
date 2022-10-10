#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers.models.bert import BertModel, BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.electra import ElectraModel, ElectraConfig
from torch.cuda.amp import autocast

from src import logger
import json


class MaskedLM(nn.Module):
    "Bert for mlm, 增加了resnet mode"

    def __init__(self,
                 pretrain_model_type,
                 pretrained_model_dir,
                 dropout,
                 res_mode=True,
                 only_config=False):
        super(MaskedLM, self).__init__()
        self.res_mode = res_mode
        model_types = (
            'bert',
            'roberta',
            'macbert',
            'electra',
        )
        assert pretrain_model_type in model_types, "pretrain_model_type should in {}".format(
            str(model_types))
        if pretrain_model_type in ['electra']:
            if not only_config:
                self.base_model = ElectraModel.from_pretrained(
                    pretrained_model_dir)
            else:
                config_fp = '{}/config.json'.format(pretrained_model_dir)
                config = json.load(open(config_fp, 'r', encoding='utf-8'))
                config = ElectraConfig(**config)
                self.base_model = ElectraModel(config)

        else:
            if not only_config:
                self.base_model = BertModel.from_pretrained(
                    pretrained_model_dir)
            else:
                config_fp = '{}/config.json'.format(pretrained_model_dir)
                config = json.load(open(config_fp, 'r', encoding='utf-8'))
                config = BertConfig(**config)
                self.base_model = BertModel(config)
        logger.info('Base model loaded.')
        self.dropout = nn.Dropout(dropout)
        if self.res_mode:
            self.transform = nn.Linear(self.base_model.config.hidden_size * 2,
                                       self.base_model.config.hidden_size)
        self.classifier = BertOnlyMLMHead(self.base_model.config)

    @autocast()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.base_model.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.base_model.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.base_model.config.use_return_dict

        if self.base_model.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.base_model.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[
            2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)),
                device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,
                                         dtype=torch.long,
                                         device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.base_model.get_extended_attention_mask(
            attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.base_model.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size(
            )
            encoder_hidden_shape = (encoder_batch_size,
                                    encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape,
                                                    device=device)
            encoder_extended_attention_mask = self.base_model.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.base_model.get_head_mask(
            head_mask, self.base_model.config.num_hidden_layers)

        embedding_output = self.base_model.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.base_model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        if self.res_mode:
            sequence_output = torch.cat(
                (sequence_output - embedding_output, sequence_output), dim=-1)
            sequence_output = self.transform(sequence_output)
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.classifier(sequence_output)

        return sequence_output


class MLMBert(BertPreTrainedModel):
    "单纯使用Roberta预训练模型"

    def __init__(
        self,
        config
    ):
        super(MLMBert, self).__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()
    @autocast()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        sequence_output = self.cls(sequence_output)
        return sequence_output
