from typing import Tuple

import torch
from src.deeplearning.loss.focal_loss import FocalCELoss
from src.deeplearning.loss.sim_loss import sent_token_simloss
from torch import device, nn
from torch.cuda.amp import autocast
from transformers.models.bert.modeling_bert import (BertConfig, BertEmbeddings,
                                                    BertEncoder,
                                                    BertForMaskedLM,
                                                    BertLMPredictionHead,
                                                    BertModel,
                                                    BertPreTrainedModel)
from transformers.models.gpt2.modeling_gpt2 import (GPT2Block, GPT2Config,
                                                    GPT2LMHeadModel, GPT2Model,GPT2Attention,
                                                    GPT2PreTrainedModel)


class ModelingUnilmCsc(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        config.num_hidden_layers = 4
        unidirectional_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.max_position_embeddings,
            n_embd=config.hidden_size,
            n_layer=4,
            n_head=config.num_attention_heads)
        bidrectional_config = config

        self.embeddings = BertEmbeddings(config)
        self.l2r_bert = nn.ModuleList(
            [GPT2Block(unidirectional_config) for _ in range(config.num_hidden_layers)])
        self.r2l_bert = nn.ModuleList(
            [GPT2Block(unidirectional_config) for _ in range(config.num_hidden_layers)])
        self.bi_bert = BertEncoder(bidrectional_config)

        self.ln_f = nn.LayerNorm(
            config.hidden_size, eps=unidirectional_config.layer_norm_epsilon)

        self.l2r_tag_projector = nn.Linear(
            config.hidden_size, config.vocab_size)
        self.r2l_tag_projector = nn.Linear(
            config.hidden_size, config.vocab_size)
        self.bi_detect_tag_projector = nn.Linear(
            config.hidden_size, 2)
        self.final_tag_projector = BertLMPredictionHead(config)
        self.mixed_feature_projector = nn.Linear(
            config.hidden_size*3, config.hidden_size)
        self.init_weights()

        self.load_criterion()

    def get_extended_attention_mask(self,
                                    attention_mask: torch.Tensor,
                                    input_shape: Tuple[int],
                                    is_decoder: bool,
                                    device: device,
                                    l2r_mask: bool = True) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)

                if l2r_mask:
                    causal_mask = seq_ids[None, None, :].repeat(
                        batch_size, seq_length, 1) <= seq_ids[None, :, None]
                else:
                    causal_mask = seq_ids[None, None, :].repeat(
                        batch_size, seq_length, 1) >= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - \
                        causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length,
                                 prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None,
                                                      :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def load_criterion(self):
        self.l2r_criterion = FocalCELoss()
        self.r2l_criterion = FocalCELoss()
        self.bi_detect_criterion = FocalCELoss(loss_labels_weights=[1, 3])
        self.focal_criterion = FocalCELoss()

    @autocast()
    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                detect_labels=None,
                correct_labels=None
                ):
        device = input_ids.device
        h = self.embeddings(input_ids=input_ids,
                            token_type_ids=token_type_ids
                            )

        unidirectional_mask = self.get_extended_attention_mask(attention_mask=attention_mask,
                                                               input_shape=input_ids.shape,
                                                               is_decoder=False,
                                                               device=device)
        l2rdirectional_mask = self.get_extended_attention_mask(attention_mask=attention_mask,
                                                               input_shape=input_ids.shape,
                                                               is_decoder=True,
                                                               device=device)
        r2ldirectional_mask = self.get_extended_attention_mask(attention_mask=attention_mask,
                                                               input_shape=input_ids.shape,
                                                               is_decoder=True,
                                                               l2r_mask=False,
                                                               device=device)

        bi_h = self.bi_bert(h, attention_mask=unidirectional_mask)[
            0]  # b, l, h
        l2r_h, r2l_h = None, None
        for block in self.l2r_bert:
            l2r_h = block(h, attention_mask=l2rdirectional_mask)[0] if l2r_h is None else block(
                l2r_h, attention_mask=l2rdirectional_mask)[0]

        for block in self.r2l_bert:
            r2l_h = block(h, attention_mask=r2ldirectional_mask)[0] if r2l_h is None else block(
                r2l_h, attention_mask=r2ldirectional_mask)[0]

        l2r_h, r2l_h = self.ln_f(l2r_h), self.ln_f(r2l_h)

        final_hidden = torch.cat((
            torch.roll(l2r_h, 1, dims=1),
            torch.roll(r2l_h, -1, dims=1),
            bi_h,
        ), dim=-1)

        final_hidden = self.mixed_feature_projector(final_hidden)

        final_hidden = self.final_tag_projector(final_hidden)

        l2r_h, r2l_h = self.l2r_tag_projector(
            l2r_h), self.r2l_tag_projector(r2l_h)

        if correct_labels is not None:

            # Shift so that tokens < n predict n
            # src: [S]  a  b c  [E]
            # trg: -1   a  b c   -1

            # l2r:
            # [S]  a  b  c
            #  a   b  c  -1

            # l2r:
            # [S]  a  b  c [E]
            #  -1 -1  a  b  c

            l2r_h = l2r_h[..., :-1, :].contiguous()
            l2r_shift_labels = correct_labels[..., 1:].contiguous()

            r2l_h = r2l_h[..., 1:, :].contiguous()
            r2l_shift_labels = correct_labels[..., :-1].contiguous()

            # Flatten the tokens

            correct_labels = correct_labels.view(-1)
            l2r_loss = self.l2r_criterion(
                l2r_h.view(-1, self.config.vocab_size), l2r_shift_labels.view(-1)
            )

            r2l_loss = self.r2l_criterion(
                r2l_h.view(-1, self.config.vocab_size),
                r2l_shift_labels.view(-1)
            )

            final_loss = self.focal_criterion(
                final_hidden.view(-1, self.config.vocab_size),
                correct_labels.view(-1)
            )

            if detect_labels is not None:

                bi_h = self.bi_detect_tag_projector(bi_h)
                bi_detect_loss = self.bi_detect_criterion(
                    bi_h.view(-1, 2),
                    detect_labels.view(-1)
                )
            # sim_loss = sent_token_simloss(
            #     y_pred=bi_h, sim_labels=detect_labels, input_ids=input_ids) if detect_labels is not None else 0

            # total_loss = sim_loss + l2r_loss + r2l_loss + final_loss
            # return final_hidden, total_loss, sim_loss, l2r_loss, r2l_loss, final_loss
            total_loss = bi_detect_loss + l2r_loss + r2l_loss + final_loss

            return final_hidden, total_loss, bi_detect_loss, l2r_loss, r2l_loss, final_loss
        else:
            return final_hidden, l2r_h, r2l_h, bi_h


class ModelingUnilmCscNew(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.gpt_model = GPT2Model.from_pretrained(
            'pretrained_model/gpt2_cn_cluesmall')
        self.bert_model = BertModel.from_pretrained(
            'pretrained_model/chinese-roberta-wwm-ext', add_pooling_layer=False)

        self.l2r_tag_projector = nn.Linear(
            config.hidden_size, config.vocab_size)
        self.bi_detect_tag_projector = nn.Linear(
            config.hidden_size, 2)
        self.final_tag_projector = BertLMPredictionHead(config)
        self.mixed_feature_projector = nn.Linear(
            config.hidden_size*2, config.hidden_size)
        self.init_weights()

        self.load_criterion()

    def get_extended_attention_mask(self,
                                    attention_mask: torch.Tensor,
                                    input_shape: Tuple[int],
                                    is_decoder: bool,
                                    device: device,
                                    l2r_mask: bool = True) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)

                if l2r_mask:
                    causal_mask = seq_ids[None, None, :].repeat(
                        batch_size, seq_length, 1) <= seq_ids[None, :, None]
                else:
                    causal_mask = seq_ids[None, None, :].repeat(
                        batch_size, seq_length, 1) >= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - \
                        causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length,
                                 prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None,
                                                      :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def load_criterion(self):
        self.l2r_criterion = FocalCELoss()
        self.r2l_criterion = FocalCELoss()
        self.bi_detect_criterion = FocalCELoss(loss_labels_weights=[1, 3])
        self.focal_criterion = FocalCELoss()

    # @autocast()
    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                detect_labels=None,
                correct_labels=None
                ):

        bi_h = self.bert_model(input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids)[0]  # b, l, h

        l2r_h = self.gpt_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

        final_hidden = torch.cat((
            torch.roll(l2r_h, 1, dims=1),
            bi_h,
        ), dim=-1)

        final_hidden = self.mixed_feature_projector(final_hidden)

        final_hidden = self.final_tag_projector(final_hidden)

        l2r_h = self.l2r_tag_projector(
            l2r_h)

        if correct_labels is not None:

            # Shift so that tokens < n predict n
            # src: [S]  a  b c  [E]
            # trg: -1   a  b c   -1

            # l2r:
            # [S]  a  b  c
            #  a   b  c  -1

            # l2r:
            # [S]  a  b  c [E]
            #  -1 -1  a  b  c

            l2r_h = l2r_h[..., :-1, :].contiguous()
            l2r_shift_labels = correct_labels[..., 1:].contiguous()

            # Flatten the tokens

            correct_labels = correct_labels.view(-1)
            l2r_loss = self.l2r_criterion(
                l2r_h.view(-1, self.config.vocab_size), l2r_shift_labels.view(-1)
            )

            final_loss = self.focal_criterion(
                final_hidden.view(-1, self.config.vocab_size),
                correct_labels.view(-1)
            )

            if detect_labels is not None:

                bi_h = self.bi_detect_tag_projector(bi_h)
                bi_detect_loss = self.bi_detect_criterion(
                    bi_h.view(-1, 2),
                    detect_labels.view(-1)
                )
            # sim_loss = sent_token_simloss(
            #     y_pred=bi_h, sim_labels=detect_labels, input_ids=input_ids) if detect_labels is not None else 0

            # total_loss = sim_loss + l2r_loss + r2l_loss + final_loss
            # return final_hidden, total_loss, sim_loss, l2r_loss, r2l_loss, final_loss
            total_loss = bi_detect_loss + l2r_loss + final_loss

            return final_hidden, total_loss, bi_detect_loss, l2r_loss, 0, final_loss
        else:
            return final_hidden, l2r_h, 0, bi_h


class ModelGpt(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.q_linear = nn.Linear(config.n_embd, config.n_embd)
        self.k_linear = nn.Linear(config.n_embd, config.n_embd)
        self.v_linear = nn.Linear(config.n_embd, config.n_embd)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.l2r_criterion = FocalCELoss()
        self.correct_criterion = FocalCELoss()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                detect_labels=None,
                correct_labels=None):

        # on train pos sample

        l2r_h = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        # compute diff between pos and neg

        input_features = self.lm_head.weight[
            input_ids].detach()[..., 1:, :]  # （b, l-1,  h）

        l2r_h = l2r_h[..., :-1, :].contiguous()

        k = torch.stack((l2r_h, input_features), dim=2)  # （b, l-1, 2, h）
        v = torch.stack((l2r_h, input_features), dim=2)

        w = torch.einsum('blmh,blnh->blmn', self.q_linear(l2r_h.unsqueeze(-2)), self.k_linear(k))            # （b, l, 1, 2）
        w = w / (float(v.size(-1)) ** 0.5)  
        w = nn.Softmax(dim=-1)(w)

        v = torch.einsum('blmn,blnh->blmh', w,
                         self.v_linear(v)).squeeze(dim=-2)
        correct_h = self.lm_head(v)
        
        
        l2r_h = self.lm_head(l2r_h)

        

        if correct_labels is not None:

            # Shift so that tokens < n predict n
            # src: [S]  a  b c  [E]
            # trg: -1   a  b c   -1

            # l2r:
            # [S]  a  b  c
            #  a   b  c  -1

            # l2r:
            # [S]  a  b  c [E]
            #  -1 -1  a  b  c

            l2r_shift_labels = correct_labels[..., 1:].contiguous()

            # Flatten the tokens

            correct_labels = correct_labels.view(-1)
            l2r_loss = self.l2r_criterion(
                l2r_h.view(-1, self.config.vocab_size), l2r_shift_labels.view(-1)
            )

            correct_loss = self.correct_criterion(
                correct_h.view(-1,
                                self.config.vocab_size), l2r_shift_labels.view(-1)
            )
            total_loss = l2r_loss + correct_loss
            return correct_h, l2r_h,  l2r_loss, correct_loss, total_loss
            # return l2r_h, l2r_h,  l2r_loss, l2r_loss, l2r_loss
        return correct_h, l2r_h



class ModelingUnilmCscnobi(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        config.num_hidden_layers = 6
        unidirectional_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.max_position_embeddings,
            n_embd=config.hidden_size,
            n_layer=6,
            n_head=config.num_attention_heads)
        bidrectional_config = config

        self.embeddings = BertEmbeddings(config)
        self.l2r_bert = nn.ModuleList(
            [GPT2Block(unidirectional_config) for _ in range(config.num_hidden_layers)])
        self.r2l_bert = nn.ModuleList(
            [GPT2Block(unidirectional_config) for _ in range(config.num_hidden_layers)])
        self.bi_bert = BertEncoder(bidrectional_config)
        
        self.ln_f = nn.LayerNorm(
            config.hidden_size, eps=unidirectional_config.layer_norm_epsilon)

        # self.l2r_tag_projector = nn.Linear(
        #     config.hidden_size, config.vocab_size)
        # self.r2l_tag_projector = nn.Linear(
        #     config.hidden_size, config.vocab_size)
        self.bi_detect_tag_projector = nn.Linear(
            config.hidden_size, 2)
        self.final_tag_projector = BertLMPredictionHead(config)
        self.mixed_feature_projector = nn.Linear(
            config.hidden_size*3, config.hidden_size)
        self.init_weights()

        self.load_criterion()

    def get_extended_attention_mask(self,
                                    attention_mask: torch.Tensor,
                                    input_shape: Tuple[int],
                                    is_decoder: bool,
                                    device: device,
                                    l2r_mask: bool = True) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)

                if l2r_mask:
                    causal_mask = seq_ids[None, None, :].repeat(
                        batch_size, seq_length, 1) <= seq_ids[None, :, None]
                else:
                    causal_mask = seq_ids[None, None, :].repeat(
                        batch_size, seq_length, 1) >= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - \
                        causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length,
                                 prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None,
                                                      :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def load_criterion(self):
        self.l2r_criterion = FocalCELoss()
        self.r2l_criterion = FocalCELoss()
        # self.bi_detect_criterion = FocalCELoss(loss_labels_weights=[1, 3])
        self.focal_criterion = FocalCELoss()

    @autocast()
    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                detect_labels=None,
                correct_labels=None
                ):
        device = input_ids.device
        h = self.embeddings(input_ids=input_ids,
                            token_type_ids=token_type_ids
                            )

        unidirectional_mask = self.get_extended_attention_mask(attention_mask=attention_mask,
                                                               input_shape=input_ids.shape,
                                                               is_decoder=False,
                                                               device=device)
        l2rdirectional_mask = self.get_extended_attention_mask(attention_mask=attention_mask,
                                                               input_shape=input_ids.shape,
                                                               is_decoder=True,
                                                               device=device)
        r2ldirectional_mask = self.get_extended_attention_mask(attention_mask=attention_mask,
                                                               input_shape=input_ids.shape,
                                                               is_decoder=True,
                                                               l2r_mask=False,
                                                               device=device)

        bi_h = self.bi_bert(h, attention_mask=unidirectional_mask)[
            0]  # b, l, h

   
        l2r_h, r2l_h = None, None
        for block in self.l2r_bert:
            l2r_h = block(h, attention_mask=l2rdirectional_mask)[0] if l2r_h is None else block(
                l2r_h, attention_mask=l2rdirectional_mask)[0]

        for block in self.r2l_bert:
            r2l_h = block(h, attention_mask=r2ldirectional_mask)[0] if r2l_h is None else block(
                r2l_h, attention_mask=r2ldirectional_mask)[0]

        l2r_h, r2l_h = self.ln_f(l2r_h), self.ln_f(r2l_h)

        
        l2r_h, r2l_h = torch.roll(l2r_h, 1, dims=1), torch.roll(r2l_h, -1, dims=1)
        final_hidden = torch.cat((
            l2r_h,
            r2l_h,
            bi_h
        ), dim=-1)

        final_hidden = self.mixed_feature_projector(final_hidden)

        final_hidden = self.final_tag_projector(final_hidden)

        l2r_h, r2l_h, bi_h = self.final_tag_projector(
            l2r_h), self.final_tag_projector(r2l_h), self.final_tag_projector(bi_h)

        if correct_labels is not None:

            # Shift so that tokens < n predict n
            # src: [S]  a  b c  [E]
            # trg: -1   a  b c   -1

            # l2r:
            # [S]  a  b  c
            #  a   b  c  -1

            # l2r:
            # [S]  a  b  c [E]
            #  -1 -1  a  b  c

            # l2r_h = l2r_h[..., :-1, :].contiguous()
            # l2r_shift_labels = correct_labels[..., 1:].contiguous()

            # r2l_h = r2l_h[..., 1:, :].contiguous()
            # r2l_shift_labels = correct_labels[..., :-1].contiguous()

            # Flatten the tokens

            correct_labels = correct_labels.view(-1)
            l2r_loss = self.l2r_criterion(
                l2r_h.view(-1, self.config.vocab_size), correct_labels
            )

            r2l_loss = self.r2l_criterion(
                r2l_h.view(-1, self.config.vocab_size),
                correct_labels
            )
            
            bi_loss = self.r2l_criterion(
                bi_h.view(-1, self.config.vocab_size),
                correct_labels
            )

            final_loss = self.focal_criterion(
                final_hidden.view(-1, self.config.vocab_size),
                correct_labels
            )

            # if detect_labels is not None:

            #     bi_h = self.bi_detect_tag_projector(bi_h)
            #     bi_detect_loss = self.bi_detect_criterion(
            #         bi_h.view(-1, 2),
            #         detect_labels.view(-1)
            #     )
            # sim_loss = sent_token_simloss(
            #     y_pred=bi_h, sim_labels=detect_labels, input_ids=input_ids) if detect_labels is not None else 0

            # total_loss = sim_loss + l2r_loss + r2l_loss + final_loss
            # return final_hidden, total_loss, sim_loss, l2r_loss, r2l_loss, final_loss
            total_loss =  l2r_loss + r2l_loss + final_loss + bi_loss

            return final_hidden, final_loss, l2r_loss, r2l_loss, total_loss
        else:
            return final_hidden, l2r_h, r2l_h





class ModelingUnilmCscBiAtt(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        config.num_hidden_layers = 6
        unidirectional_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.max_position_embeddings,
            n_embd=config.hidden_size,
            n_layer=6,
            n_head=config.num_attention_heads)
        bidrectional_config = config

        self.embeddings = BertEmbeddings(config)
        self.l2r_bert = nn.ModuleList(
            [GPT2Block(unidirectional_config) for _ in range(config.num_hidden_layers)])
        self.r2l_bert = nn.ModuleList(
            [GPT2Block(unidirectional_config) for _ in range(config.num_hidden_layers)])
        self.bi_bert = BertEncoder(bidrectional_config)
        self.atten = GPT2Attention(GPT2Config(vocab_size=config.vocab_size,
                                              ),
                                   is_cross_attention=True)
        self.ln_f = nn.LayerNorm(
            config.hidden_size, eps=unidirectional_config.layer_norm_epsilon)

        # self.l2r_tag_projector = nn.Linear(
        #     config.hidden_size, config.vocab_size)
        # self.r2l_tag_projector = nn.Linear(
        #     config.hidden_size, config.vocab_size)
        self.bi_detect_tag_projector = nn.Linear(
            config.hidden_size, 2)
        self.final_tag_projector = BertLMPredictionHead(config)
        # self.mixed_feature_projector = nn.Linear(
        #     config.hidden_size*3, config.hidden_size)
        self.init_weights()

        self.load_criterion()

    def get_extended_attention_mask(self,
                                    attention_mask: torch.Tensor,
                                    input_shape: Tuple[int],
                                    is_decoder: bool,
                                    device: device,
                                    l2r_mask: bool = True) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)

                if l2r_mask:
                    causal_mask = seq_ids[None, None, :].repeat(
                        batch_size, seq_length, 1) <= seq_ids[None, :, None]
                else:
                    causal_mask = seq_ids[None, None, :].repeat(
                        batch_size, seq_length, 1) >= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - \
                        causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length,
                                 prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None,
                                                      :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def load_criterion(self):
        self.l2r_criterion = FocalCELoss()
        self.r2l_criterion = FocalCELoss()
        # self.bi_detect_criterion = FocalCELoss(loss_labels_weights=[1, 3])
        self.focal_criterion = FocalCELoss()

    # @autocast()
    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                detect_labels=None,
                correct_labels=None
                ):
        device = input_ids.device
        h = self.embeddings(input_ids=input_ids,
                            token_type_ids=token_type_ids
                            )

        unidirectional_mask = self.get_extended_attention_mask(attention_mask=attention_mask,
                                                               input_shape=input_ids.shape,
                                                               is_decoder=False,
                                                               device=device)
        l2rdirectional_mask = self.get_extended_attention_mask(attention_mask=attention_mask,
                                                               input_shape=input_ids.shape,
                                                               is_decoder=True,
                                                               device=device)
        r2ldirectional_mask = self.get_extended_attention_mask(attention_mask=attention_mask,
                                                               input_shape=input_ids.shape,
                                                               is_decoder=True,
                                                               l2r_mask=False,
                                                               device=device)

        bi_h = self.bi_bert(h, attention_mask=unidirectional_mask)[
            0]  # b, l, h

   
        l2r_h, r2l_h = None, None
        for block in self.l2r_bert:
            l2r_h = block(h, attention_mask=l2rdirectional_mask)[0] if l2r_h is None else block(
                l2r_h, attention_mask=l2rdirectional_mask)[0]

        for block in self.r2l_bert:
            r2l_h = block(h, attention_mask=r2ldirectional_mask)[0] if r2l_h is None else block(
                r2l_h, attention_mask=r2ldirectional_mask)[0]

        l2r_h, r2l_h = self.ln_f(l2r_h), self.ln_f(r2l_h)

        
        l2r_h, r2l_h = torch.roll(l2r_h, 1, dims=1), torch.roll(r2l_h, -1, dims=1)
        
        
        final_hidden = l2r_h + r2l_h
        
        final_hidden = self.atten(hidden_states=final_hidden, 
                                  attention_mask=unidirectional_mask,
                                  encoder_attention_mask =unidirectional_mask,
                                  encoder_hidden_states=bi_h)[0] # attn_output, attn_weights
        
        
        # final_hidden = torch.cat((
        #     l2r_h,
        #     r2l_h,
        #     final_hidden
        # ), dim=-1)

        # final_hidden = self.mixed_feature_projector(final_hidden)

        final_hidden = self.final_tag_projector(final_hidden)

        l2r_h, r2l_h, bi_h = self.final_tag_projector(
            l2r_h), self.final_tag_projector(r2l_h), self.final_tag_projector(bi_h)

        if correct_labels is not None:

            # Shift so that tokens < n predict n
            # src: [S]  a  b c  [E]
            # trg: -1   a  b c   -1

            # l2r:
            # [S]  a  b  c
            #  a   b  c  -1

            # l2r:
            # [S]  a  b  c [E]
            #  -1 -1  a  b  c

            # l2r_h = l2r_h[..., :-1, :].contiguous()
            # l2r_shift_labels = correct_labels[..., 1:].contiguous()

            # r2l_h = r2l_h[..., 1:, :].contiguous()
            # r2l_shift_labels = correct_labels[..., :-1].contiguous()

            # Flatten the tokens

            correct_labels = correct_labels.view(-1)
            l2r_loss = self.l2r_criterion(
                l2r_h.view(-1, self.config.vocab_size), correct_labels
            )

            r2l_loss = self.r2l_criterion(
                r2l_h.view(-1, self.config.vocab_size),
                correct_labels
            )
            
            bi_loss = self.r2l_criterion(
                bi_h.view(-1, self.config.vocab_size),
                correct_labels
            )

            final_loss = self.focal_criterion(
                final_hidden.view(-1, self.config.vocab_size),
                correct_labels
            )

            # if detect_labels is not None:

            #     bi_h = self.bi_detect_tag_projector(bi_h)
            #     bi_detect_loss = self.bi_detect_criterion(
            #         bi_h.view(-1, 2),
            #         detect_labels.view(-1)
            #     )
            # sim_loss = sent_token_simloss(
            #     y_pred=bi_h, sim_labels=detect_labels, input_ids=input_ids) if detect_labels is not None else 0

            # total_loss = sim_loss + l2r_loss + r2l_loss + final_loss
            # return final_hidden, total_loss, sim_loss, l2r_loss, r2l_loss, final_loss
            total_loss =  l2r_loss + r2l_loss + final_loss + bi_loss

            return final_hidden, final_loss, l2r_loss, r2l_loss, total_loss
        else:
            return final_hidden, l2r_h, r2l_h


        
def build_dummpy_inputs():
    inputs = {}
    inputs['input_ids'] = torch.LongTensor(
        torch.randint(low=1, high=10, size=(2, 6)))
    inputs['attention_mask'] = torch.randint(low=0, high=2, size=(2, 6)).long()

    inputs['detect_labels'] = torch.randint(low=0, high=2, size=(2, 6)).long()
    inputs['correct_labels'] = torch.randint(
        low=0, high=100, size=(2, 6)).long()
    inputs['token_type_ids'] = torch.zeros(size=(2, 6)).long()
    return inputs


if __name__ == '__main__':
    inputs = build_dummpy_inputs()
    
    model = ModelingUnilmCscBiAtt.from_pretrained('pretrained_model/gpt2_cn_cluesmall')

    r = model(**inputs)
    print('end')
