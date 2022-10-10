#!/usr/bin/env python
# -*- coding: utf-8 -*-


from functools import reduce
import os
import numpy as np
import torch
from src import logger
from src.deeplearning.modeling.modeling_realise import SpellBertPho2ResArch3
from src.utils.data_helper import SPACE_SIGNS, replace_punc_for_bert, replace_punc_for_bert_keep_space, SPACE_SIGNS_2
from src.utils.realise.pinyin_util import Pinyin2
from transformers.models.bert import BertTokenizer
import onnxruntime as rt  
import time


class PredictorRealise:
    def __init__(
        self,
        in_model_dir,
        max_len_char_pinyin=7,
        use_cuda=True,
        cuda_id=None,

    ):
        self.cuda_id = cuda_id
        self.in_model_dir = in_model_dir
        self.model = SpellBertPho2ResArch3.from_pretrained(
            in_model_dir)

        logger.info('model loaded from dir {}'.format(
            self.in_model_dir))
        self.use_cuda = use_cuda
        if self.use_cuda and torch.cuda.is_available():
            if self.cuda_id is not None:
                torch.cuda.set_device(self.cuda_id)
            self.model.cuda()
            self.model.half()
        self.model.eval()
        self.max_len_char_pinyin = max_len_char_pinyin
        self.pho2_convertor = Pinyin2()
        self.tokenizer = BertTokenizer.from_pretrained(in_model_dir)

    def tokenize_inputs(self, texts, return_tensors=None):
        "预测tokenize, 按batch texts中最大的文本长度来pad, realise只需要input id, mask, length"

        cls_id, sep_id, pad_id, unk_id = self.tokenizer.vocab['[CLS]'], self.tokenizer.vocab[
            '[SEP]'], self.tokenizer.vocab['[PAD]'], self.tokenizer.vocab['[UNK]']
        input_ids, attention_mask, length = [], [], []
        max_len = max([len(text) for text in texts]) + 2  # 注意+2

        for text in texts:
            true_input_id = [self.tokenizer.vocab.get(
                c, unk_id) for c in text][:max_len-2]
            pad_len = (max_len-len(true_input_id)-2)
            input_id = [cls_id] + true_input_id + [sep_id] + [pad_id] * pad_len

            a_mask = [1] * (len(true_input_id) + 2) + [0] * pad_len
            input_ids.append(input_id)
            attention_mask.append(a_mask)
            length.append(sum(a_mask))

        if return_tensors == 'pt':
            return {'input_ids': torch.LongTensor(input_ids), 'attention_mask': torch.LongTensor(attention_mask), 'length': torch.LongTensor(length)}
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'length': length}

    @torch.no_grad()
    def predict(self, texts, return_topk=1, batch_size=32):
        """ seq2label
        Args:
            texts (list): 
            ignore_eng: 是否忽略英文检错
        Returns:
            List[tuple]: [ ('中', [ ('中', 0.9977), ('众', 0.0023)] ) ],
        """
        if isinstance(texts, str):
            texts_inputs = [texts]
        else:
            texts_inputs = texts
        outputs = []
        for start_idx in range(0, len(texts_inputs), batch_size):
            batch_texts = texts_inputs[start_idx:start_idx +
                                       batch_size]
            # 将一些bert词典中不存在的 中文符号 转换为 英文符号
            # 加空格让bert对英文分字
            batch_char_based_texts = [
                replace_punc_for_bert(text)
                for text in batch_texts
            ]
            inputs = self.tokenize_inputs(batch_char_based_texts)
            input_chars = self.tokenizer.convert_ids_to_tokens(
                reduce(lambda a, b: a+b, inputs['input_ids']))
            phos_idx, phos_lens = self.pho2_convertor.convert(
                input_chars, self.max_len_char_pinyin)

            inputs['pho_idx'] = phos_idx
            inputs['pho_lens'] = torch.LongTensor(phos_lens)
            inputs['input_ids'] = torch.LongTensor(inputs['input_ids'])
            inputs['attention_mask'] = torch.LongTensor(
                inputs['attention_mask'])
            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                inputs['pho_idx'] = inputs['pho_idx'].cuda()
                inputs['pho_lens'] = inputs['pho_lens'].cuda()

            d_preds, c_preds, pinyin_preds = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pho_idx=inputs['pho_idx'],
                pho_lens=inputs['pho_lens']
            )
            
            
            for idx, length in enumerate(inputs['length']):
                # 逐条处理
                true_idx = range(1, length - 1)
                pred = c_preds[idx, true_idx, ...]

                pred_h = torch.softmax(pred, dim=-1)

                pred_h, pred = pred_h.topk(k=return_topk,
                                           dim=-1,
                                           largest=True,
                                           sorted=True)  # logit, idx
                pred_h = pred_h.tolist()

                pred = [self.tokenizer.convert_ids_to_tokens(
                    x) for x in pred]
                pred = list(list(zip(*ele)) for ele in zip(pred, pred_h))
                origin_text = batch_texts[idx]
                # 还原空格

                [pred.insert(i, [(v, 1.0)]) for i, v in enumerate(
                    origin_text) if v in SPACE_SIGNS]
                outputs.append(list(zip(origin_text, pred)))
        return outputs

    def convert_to_jit(self):
        
        
        # self.model.cpu()
        self.model.bert = torch.jit.trace(self.model.bert, (torch.randint(1, 5, (3, 122)).cuda(),
                                                            torch.randint(1, 5, (3, 122)).cuda()), strict=False)
        
        
        
        logger.info('bert layer -> jit format')

    