import os
from copy import deepcopy

import numpy as np
import opencc
import torch
from PIL import ImageFont
from src.deeplearning.modeling.realise_sub_char_cnn import CharResNet, CharResNet1
from src.utils.realise.pinyin_util import _is_chinese_char, pho2_convertor
from torch import nn
from transformers.models.bert import BertModel, BertPreTrainedModel
from torch.cuda.amp import autocast



class SpellBertPho2ResArch3(BertPreTrainedModel):
    def __init__(self, config):
        super(SpellBertPho2ResArch3, self).__init__(config)
        self.config = config

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)

        self.pho_embeddings = nn.Embedding(pho2_convertor.get_pho_size(),
                                           config.hidden_size,
                                           padding_idx=0)
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        pho_config = deepcopy(config)
        pho_config.num_hidden_layers = 4
        self.pho_model = BertModel(pho_config)

        if self.config.num_fonts == 1:
            self.char_images = nn.Embedding(config.vocab_size, 1024)
            self.char_images.weight.requires_grad = False
        else:
            self.char_images_multifonts = torch.nn.Parameter(
                torch.rand(config.vocab_size, self.config.num_fonts, 32, 32))
            self.char_images_multifonts.requires_grad = False

        if config.image_model_type == 0:
            self.resnet = CharResNet(in_channels=self.config.num_fonts)
        elif config.image_model_type == 1:
            self.resnet = CharResNet1()
        else:
            raise NotImplementedError('invalid image_model_type %d' %
                                      config.image_model_type)
        self.resnet_layernorm = nn.LayerNorm(config.hidden_size,
                                             eps=config.num_labels)

        self.gate_net = nn.Linear(4 * config.hidden_size, 3)

        out_config = deepcopy(config)
        out_config.num_hidden_layers = 3
        self.output_block = BertModel(out_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        if config.add_detect_task:
            self.detect_cls = nn.Linear(config.hidden_size, config.type_vocab_size)
        if config.add_pinyin_task: 
            self.pinyin_cls = nn.Linear(config.hidden_size, config.type_pinyin_vocab_size)
        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    def build_glyce_embed(self, vocab_dir, font_path, font_size=32):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [s.strip() for s in f]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) != 1 or (not _is_chinese_char(ord(char))):
                char_images.append(
                    np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(
                image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros(
                    (font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0],
                           offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images -
                       np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).reshape(
            char_images.shape[0], -1)
        assert char_images.shape == (self.config.vocab_size, 1024)
        self.char_images.weight.data.copy_(char_images)

 
    def build_glyce_embed_multifonts(self,
                                     vocab_dir,
                                     num_fonts,
                                     use_traditional_font,
                                     font_size=32):
        font_paths = [
            ('src/data/font_images/simhei.ttf', False),
            ('src/data/font_images/xiaozhuan.ttf', False),
            ('src/data/font_images/simhei.ttf', True),
        ]
        font_paths = font_paths[:num_fonts]
        if use_traditional_font:
            font_paths = font_paths[:-1]
            font_paths.append(('src/data/font_images/simhei.ttf', True))
            self.converter = opencc.OpenCC('s2t.json')

        images_list = []
        for font_path, use_traditional in font_paths:
            images = self.build_glyce_embed_onefont(
                vocab_dir=vocab_dir,
                font_path=font_path,
                font_size=font_size,
                use_traditional=use_traditional,
            )
            images_list.append(images)

        char_images = torch.stack(images_list, dim=1).contiguous()
        self.char_images_multifonts.data.copy_(char_images)

     
    def build_glyce_embed_onefont(self, vocab_dir, font_path, font_size,
                                  use_traditional):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path) as f:
            vocab = [s.strip() for s in f.readlines()]
        if use_traditional:
            vocab = [
                self.converter.convert(c) if len(c) == 1 else c for c in vocab
            ]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) > 1:
                char_images.append(
                    np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(
                image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros(
                    (font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0],
                           offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images -
                       np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).contiguous()
        return char_images

    @staticmethod
    def build_batch(batch, tokenizer):
        src_idx = batch['input_ids'].flatten().tolist()
        chars = tokenizer.convert_ids_to_tokens(src_idx)
        pho_idx, pho_lens = pho2_convertor.convert(chars)
        batch['pho_idx'] = pho_idx
        batch['pho_lens'] = pho_lens
        return batch
    
    @autocast()
    def forward(self, input_ids, attention_mask, pho_idx, pho_lens):
        pho_idx = pho_idx.view(-1, 7)
        pho_lens = pho_lens.flatten().tolist()
        
        # label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

        input_shape = input_ids.size()

        # bert_hiddens = self.bert(input_ids, attention_mask=attention_mask)[0]
        bert_hiddens = self.bert(input_ids, attention_mask=attention_mask)['last_hidden_state']

        pho_embeddings = self.pho_embeddings(pho_idx)
        pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        self.pho_gru.flatten_parameters()
        _, pho_hiddens = self.pho_gru(pho_embeddings)
        
        pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0],
                                                     input_shape[1],
                                                     -1).contiguous()
        pho_hiddens = self.pho_model(inputs_embeds=pho_hiddens,
                                     attention_mask=attention_mask)[0]

        src_idxs = input_ids.view(-1)

        if self.config.num_fonts == 1:
            images = self.char_images(src_idxs).reshape(
                src_idxs.shape[0], 1, 32, 32).contiguous()
        else:
            images = self.char_images_multifonts.index_select(dim=0,
                                                              index=src_idxs)

        res_hiddens = self.resnet(images)
        res_hiddens = res_hiddens.reshape(input_shape[0], input_shape[1],
                                          -1).contiguous()
        res_hiddens = self.resnet_layernorm(res_hiddens)

        bert_hiddens_mean = (bert_hiddens * attention_mask.to(
            torch.float).unsqueeze(2)).sum(dim=1) / attention_mask.to(
                torch.float).sum(dim=1, keepdim=True)
        bert_hiddens_mean = bert_hiddens_mean.unsqueeze(1).expand(
            -1, bert_hiddens.size(1), -1)

        concated_outputs = torch.cat(
            (bert_hiddens, pho_hiddens, res_hiddens, bert_hiddens_mean),
            dim=-1)
        gated_values = self.gate_net(concated_outputs)
        # B * S * 3
        g0 = torch.sigmoid(gated_values[:, :, 0].unsqueeze(-1))
        g1 = torch.sigmoid(gated_values[:, :, 1].unsqueeze(-1))
        g2 = torch.sigmoid(gated_values[:, :, 2].unsqueeze(-1))

        hiddens = g0 * bert_hiddens + g1 * pho_hiddens + g2 * res_hiddens

        outputs = self.output_block(inputs_embeds=hiddens,
                                    position_ids=torch.zeros(
                                        input_ids.size(),
                                        dtype=torch.long,
                                        device=input_ids.device),
                                    attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        
        d_logits = self.detect_cls(sequence_output) if self.config.add_detect_task else None
        
        pinyin_logits = self.pinyin_cls(sequence_output) if self.config.add_pinyin_task else None
        
        c_logits = self.classifier(sequence_output)
        
        return d_logits, c_logits, pinyin_logits
