import torch
from src.deeplearning.loss.focal_loss import FocalCELoss
from torch import nn
from transformers.modeling_outputs import (BaseModelOutput, Seq2SeqLMOutput,
                                           Seq2SeqModelOutput)
from transformers.models.bart.modeling_bart import (BartConfig, BartDecoder,
                                                    BartEncoder, BartModel,
                                                    BartPretrainedModel)
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertOnlyMLMHead



def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids




class Detector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projector1 = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.activation_function, str):
            self.transform_act_fn = ACT2FN[config.activation_function]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        
        self.projector2 = nn.Linear(config.hidden_size, config.detector_label_num, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.detector_label_num))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.projector2.bias = self.bias

    def forward(self, sequence_output):
        sequence_output = self.projector1(sequence_output)
        sequence_output = self.transform_act_fn(sequence_output)
        sequence_output = self.LayerNorm(sequence_output)
        sequence_output = self.projector2(sequence_output)
        return sequence_output

class ModelingCtcBart(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        self.activation_fn = ACT2FN[config.activation_function]
        
        self.detector = Detector(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)

        self.init_weights()
        
        self.detector_loss_fct = FocalCELoss(loss_labels_weights=[1, 3],ignore_index=-100)
        self.corrector_loss_fct = FocalCELoss(ignore_index=-100)
    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared
    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        
    @staticmethod
    def build_dummy_inputs():
        inputs = {}
        inputs['input_ids'] = torch.randint(0, 10, (2, 8)).long()
        inputs['attention_mask'] = torch.ones(2, 8).long()
        inputs['labels'] = torch.ones((2, 8)).long()
        inputs['detector_labels'] = torch.ones((2, 8)).long()
        return inputs
        
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        detector_labels=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                
        elif decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        
        # concat cls and seq hidden
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[:,0,:].unsqueeze(dim=1)+encoder_outputs.last_hidden_state # (b, l, 768*2)
        
        
        
        detector_hiddens = self.detector(
            encoder_outputs.last_hidden_state
            )
      
        if detector_labels is not None:
            detector_loss = self.detector_loss_fct(detector_hiddens.view(-1, self.config.detector_label_num), detector_labels.view(-1))
        else:
            detector_loss = None
        
        # encoder_outputs[0] = e
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
    
        outputs = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        corrector_loss = None
        if labels is not None:
            
            corrector_loss = self.corrector_loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            if detector_labels is not None:
                corrector_loss += detector_loss


        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((corrector_loss,) + output) if corrector_loss is not None else output

        return Seq2SeqLMOutput(
            loss=corrector_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


if __name__ == "__main__":
    model_dir = 'pretrained_model/bart-base-chinese-cluecorpussmall'
    model = ModelingCtcBart.from_pretrained(model_dir)
    inputs = ModelingCtcBart.build_dummy_inputs()
    out = model(**inputs)
    print(out)
    