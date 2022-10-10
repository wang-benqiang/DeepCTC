#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from functools import reduce

import torch
from src import logger
from src.deeplearning.modeling.modeling_mlm_new import ModelMlm
from transformers.models.bert import BertTokenizer
from src.utils.data_helper import include_cn
from src.utils.data_helper import replace_punc_for_bert, SPACE_SIGNS
import numpy as np


class PredictorMlm:
    def __init__(
            self,
            pretrained_model_dir,
            use_cuda=True,
            cuda_id=None,
    ):

        self.pretrained_model_dir = pretrained_model_dir
        self.use_cuda = use_cuda

        self.eng_digit_re = re.compile(r'[a-zA-Z]|[0-9]')

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_dir)
        self.model = self.load_model()
        if self.use_cuda and torch.cuda.is_available():
            if cuda_id is not None:
                torch.cuda.set_device(cuda_id)
            self.model = self.model.cuda()
            self.model.half()

    def load_model(self):

        model = ModelMlm.from_pretrained(self.pretrained_model_dir)
        model.eval()
        logger.info('ckpt loaded: {}'.format(self.pretrained_model_dir))
        return model

    def load_config(self):
        config_fp = "{}/train_config.json".format(self.model_dir)
        config = json.load(open(config_fp, 'r', encoding='utf-8'))
        return config

    def tokenize_inputs(self, texts, return_tensors=None):
        """预测tokenize, 按batch texts中最大的文本长度来pad

        Args:
            texts (_type_): [['你','[MASK]'], ...]
            return_tensors (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        cls_id, sep_id, pad_id, unk_id = self.tokenizer.vocab['[CLS]'], self.tokenizer.vocab[
            '[SEP]'], self.tokenizer.vocab['[PAD]'], self.tokenizer.vocab['[UNK]']
        input_ids, attention_mask, token_type_ids = [], [], []
        max_len = max([len(text) for text in texts]) + 2  # 注意+2

        for char_list in texts:
            true_input_id = [self.tokenizer.vocab.get(
                c, unk_id) for c in char_list][:max_len-2]
            pad_len = (max_len-len(true_input_id)-2)
            input_id = [cls_id] + true_input_id + [sep_id] + [pad_id] * pad_len
            a_mask = [1] * (len(true_input_id) + 2) + [0] * pad_len
            token_type_id = [0] * max_len
            input_ids.append(input_id)
            attention_mask.append(a_mask)
            token_type_ids.append(token_type_id)

        if return_tensors == 'pt':
            return {'input_ids': torch.LongTensor(input_ids),
                    'attention_mask': torch.LongTensor(attention_mask),
                    'token_type_ids': torch.LongTensor(token_type_ids),
                    }
        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                }

    @torch.no_grad()
    def predict4ms(
        self,
        texts,
        only_mask=True,
        return_topk=1,
        batch_size=32,
    ):
        """老版本少字能力用的函数

        Args:
            texts ([list]):  ['我今天吃[MASK]了', '我今天吃[MASK]了']
            only_mask (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [[('你','你'), ('[MASK]', '好')], ....]
        """
        outputs = []
        if isinstance(texts, str):
            texts = [texts]

        texts_for_bert_input = [
            ' '.join(replace_punc_for_bert(text)).replace(
                '[ M A S K ]', '[MASK]') for text in texts
        ]

        for start_idx in range(0, len(texts), batch_size):
            batch_outputs = []
            batch_texts = texts[start_idx:start_idx + batch_size]
            batch_texts_for_bert_input = texts_for_bert_input[start_idx:start_idx + batch_size]
            batch_texts_for_bert_input = [x.split(' ') for x in batch_texts_for_bert_input]
            inputs = self.tokenize_inputs(batch_texts_for_bert_input,
                                          return_tensors='pt',
                                          )
            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

            preds, loss = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
            )

            for idx, origin_text in enumerate(batch_texts):

                if only_mask:
                    # MASK idx不能从tokenize后匹配, 因为tokenize后会过滤空格
                    pred_mask_idx = np.argwhere(inputs['input_ids'][idx].cpu(
                    ).numpy() == self.tokenizer.mask_token_id).reshape(-1)

                    if len(pred_mask_idx) == 0:
                        # 如果没有mask
                        batch_outputs.append(
                            list(zip(origin_text, origin_text)))
                        continue
                    pred = preds[idx, pred_mask_idx, ...]
                else:
                    pred = preds[idx]

                pred = torch.softmax(pred, dim=-1)

                pred_h, mask_pred = pred.topk(return_topk,
                                              dim=-1,
                                              largest=True,
                                              sorted=True)
                pred_h = pred_h.tolist()
                mask_pred = list(
                    map(
                        lambda x: self.tokenizer.convert_ids_to_tokens(
                            x),
                        mask_pred
                    )
                )

                mask_pred = list(list(zip(*ele))
                                 for ele in zip(mask_pred, pred_h))

                if only_mask:
                    origin_text = self.str_with_mask_to_list(origin_text)
                    # MASK -> token
                    mask_pred = iter(mask_pred)
                    trg = [
                        ori_chr if ori_chr != '[MASK]' else next(mask_pred)
                        for ori_chr in origin_text
                    ]

                else:
                    # 如果有空格的话, 对输出的pred插入空格
                    [mask_pred.insert(i, [(v, 1.0)])
                     for i, v in enumerate(origin_text) if v in SPACE_SIGNS]
                    trg = mask_pred[1:-1]
                batch_outputs.append(list(zip(origin_text, trg)))
            outputs.extend(batch_outputs)
        return outputs

    @torch.no_grad()
    def predict(self, texts, return_topk, batch_size=32):
        """输入文本，自动对每个中文字符打mask并预测
        Args:
            texts (_type_): _description_
        """

        outputs = []  # 最终结果

        texts_mlm_input_for_every_char = []
        for text in texts:
            # 根据一条文本，生成一组带有每个字位置mask的一组文本
            text_mlm_input_for_every_char = self.build_mask_input_from_str(
                text)
            # 合并到总的输入文本里
            texts_mlm_input_for_every_char.extend(
                text_mlm_input_for_every_char)

        # 批处理进入模型
        mlm_preds = self.predict_with_mask(
            texts_mlm_input_for_every_char, return_topk=return_topk, batch_size=batch_size)

        for text in texts:
            output = []  # 单个text的output
            for idx, char in enumerate(text):
                if include_cn(char):
                    current_char_mlm_prd = mlm_preds.pop(0)
                    # 利用pop依次再还原给每一个字的mask预测结果
                    output.append(
                        (char, current_char_mlm_prd[idx][1], current_char_mlm_prd[idx][2]))
                else:
                    # 非中文直接还原
                    output.append((char, [char], [1]))
            outputs.append(output)
        return outputs

    @torch.no_grad()
    def predict_with_mask(
        self,
        texts,
        return_topk=1,
        batch_size=32,
    ):
        """预测mask任务, 或者对所有字进行纠错

        Args:
            texts ([list]):  [['你','[MASK]']]
            only_mask (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [ [('我', [...], [...]), ('将', [...], [...])], ...]
        """
        if isinstance(texts, str):
            texts = [self.str_with_mask_to_list(texts)]

        if isinstance(texts, list) and isinstance(texts[0], str):
            texts = [self.str_with_mask_to_list(text) for text in texts]
        outputs = []
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx+batch_size]
            inputs = self.tokenize_inputs(batch_texts,
                                          return_tensors='pt')

            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

            preds, loss = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
            )

            preds = torch.softmax(preds[:, 1:, :], dim=-1)  # 从cls后面开始
            recall_top_k_probs, recall_top_k_ids = preds.topk(
                k=return_topk, dim=-1, largest=True, sorted=True)
            recall_top_k_probs = recall_top_k_probs.tolist()
            recall_top_k_ids = recall_top_k_ids.tolist()
            recall_top_k_chars = [[self.tokenizer.convert_ids_to_tokens(
                char_level) for char_level in sent_level] for sent_level in recall_top_k_ids]

            batch_outputs = [list(zip(text, top_k_char, top_k_prob)) for text, top_k_char, top_k_prob in zip(
                batch_texts, recall_top_k_chars, recall_top_k_probs)]
            outputs.extend(batch_outputs)
        return outputs

    @staticmethod
    def str_with_mask_to_list(text):
        """将带有[MASK]的字符串转成list

        Args:
            text ([type]): '这 是一个[MASK]啊 ,你的[MASK]呢'

        Returns:
            [type]: ['这', ' ', '是', '一', '个', '[MASK]', '啊', ' ', ',', '你', '的', '[MASK]', '呢']
        """
        texts = text.split('[MASK]')
        if len(texts) < 2:
            return list(text)
        return list(reduce(lambda a, b: list(a) + ['[MASK]'] + list(b), texts))

    @staticmethod
    def build_mask_input_from_str(text):
        """将文本转化成每个字都被打mask的情况
        Args:
            text (_type_): _description_
        Returns:
            _type_: _description_
        """
        text_li = list(text)
        input_list = [text_li[:idx] + ['[MASK]'] + text_li[idx + 1:]

                      for idx, char in enumerate(text_li) if include_cn(char)]

        return input_list


if __name__ == '__main__':
    
    mlm_csc_model_dir = 'pretrained_model/chinese-roberta-wwm-ext'
    mlm_model_dir = 'model/mlm_roberta_base_pretrained_2022Y03M22D12H/epoch1,ith_db1, step98274,testf1_54_05%,devf1_66_36%'
    p = PredictorMlm(
        pretrained_model_dir=mlm_csc_model_dir)
    
    
    outputs = p.predict(
        texts=['我将在学校挑选两名同意共同探讨这个方案', '我上课要吃到了'], return_topk=5)
    print(outputs)
