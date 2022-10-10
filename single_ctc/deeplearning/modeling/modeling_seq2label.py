#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from TorchCRF import CRF
from src import logger
from torch import nn
from transformers.models.electra import ElectraModel, ElectraPreTrainedModel
from transformers.activations import get_activation
from torch.cuda.amp import autocast


class Sequence2Label(ElectraPreTrainedModel):
    """合并了detector和Corrector
     "检错的话,预测对应位置的append, keep, replace, delete"
     "纠错的话,直接预测对应位置的CTC label的 1.6W标签"
    """

    def __init__(self,
                 config):
        super(Sequence2Label, self).__init__(config)
        self.config = config

        if config.append_crf:
            self.crf_layer = CRF(config.num_labels)

        self.electra = ElectraModel(config) # v1.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            config.hidden_size *
            2 if config.res_mode else config.hidden_size,
            config.num_labels)
        if config.two_cls_layer:
            logger.info('build model with two layer cls')
            self.pre_cls = nn.Linear(
                config.hidden_size *
                2 if config.res_mode else config.hidden_size,
                config.hidden_size *
                2 if config.res_mode else config.hidden_size,)

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

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
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
        extended_attention_mask: torch.Tensor = self.electra.get_extended_attention_mask(
            attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size(
            )
            encoder_hidden_shape = (encoder_batch_size,
                                    encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape,
                                                    device=device)
            encoder_extended_attention_mask = self.electra.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.electra.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.electra.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.electra.encoder(
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
        if self.config.res_mode:
            sequence_output = torch.cat((embedding_output, sequence_output),
                                        dim=-1)

        sequence_output = self.dropout(sequence_output)

        if self.config.two_cls_layer:
            sequence_output = self.pre_cls(sequence_output)
            sequence_output = get_activation(
                self.config.hidden_act)(sequence_output)
        sequence_output = self.classifier(sequence_output)

        if self.config.append_crf:
            crf_loglikelihood = self.crf_layer(
                sequence_output, self.num_labels, attention_mask)
            return crf_loglikelihood

        return sequence_output


if __name__ == "__main__":

    pretrained_dir = 'pretrained_model/electra_base_cn_discriminator'
    model = Sequence2Label.from_pretrained(pretrained_dir)
    print('1')
