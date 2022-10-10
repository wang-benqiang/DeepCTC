#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from functools import reduce

import numpy as np
import torch
from src import logger
from src.deeplearning.modeling.modeling_mlm import MLMBert
from src.utils.data_helper import SPACE_SIGNS, replace_punc_for_bert
from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
from configs.ctc_conf import CtcConf


class PredictorMlm:
    def __init__(
            self,
            pretrained_model_dir,
            pretrained_model_type='bert',
            use_cuda=True,
            cuda_id=None,
            onnx_mode=False
    ):

        self.pretrained_model_type = pretrained_model_type
        self.pretrained_model_dir = pretrained_model_dir
        self.use_cuda = use_cuda

        self.eng_digit_re = re.compile(r'[a-zA-Z]|[0-9]')
        self.cuda_id = cuda_id
        self.onnx_mode = onnx_mode
        self.tokenizer = CustomBertTokenizer.from_pretrained(
            pretrained_model_dir)
        self.model = self.load_model()

    def load_model(self):

        if CtcConf.ascend_mode:
            from src.corrector.corrector import AclLiteModel
            model = AclLiteModel(CtcConf.mlm_recall_model_fp_ascend_mode)
            logger.info('model loaded from: {}'.format(
                CtcConf.mlm_recall_model_fp_ascend_mode))
        else:

            model = MLMBert.from_pretrained(self.pretrained_model_dir)
            model.eval()
            logger.info('ckpt loaded: {}'.format(self.pretrained_model_dir))
            if self.use_cuda and torch.cuda.is_available():
                if self.cuda_id is not None:
                    torch.cuda.set_device(self.cuda_id)
                model.cuda()

                if not self.onnx_mode:
                    model = model.half()  # 半精度
            logger.info('model loaded from: {}'.format(
                self.pretrained_model_dir))
        return model

    def load_config(self):
        config_fp = "{}/train_config.json".format(self.model_dir)
        config = json.load(open(config_fp, 'r', encoding='utf-8'))
        return config

    @torch.no_grad()
    def __call__(
        self,
        texts,
        only_mask=True,
        return_topk=1,
        batch_size=32,
    ):
        """预测mask任务, 或者对所有字进行纠错

        Args:
            texts ([list]):  ['我今天吃[MASK]了', '我今天吃[MASK]了']
            only_mask (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [[('你','你'), ('[MASK]', '好')], ....]
        """
        
        if CtcConf.ascend_mode:
            return self.call_for_ascend_wrapper(texts, batch_size, return_topk=return_topk)
        outputs = []
        if isinstance(texts, str):
            texts = [texts]

        texts_for_bert_input = [
            ' '.join(replace_punc_for_bert(text)).replace(
                '[ M A S K ]', '[MASK]') for text in texts
        ]
        texts_for_bert_input =[ x.split(' ') for x in texts_for_bert_input]
        for start_idx in range(0, len(texts), batch_size):
            batch_outputs = []
            batch_texts = texts[start_idx:start_idx + batch_size]
            batch_texts_for_bert_input = texts_for_bert_input[start_idx:start_idx + batch_size]
            inputs = self.tokenizer(batch_texts_for_bert_input,
                                    return_tensors='pt',
                                    )
            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

            preds = self.model(
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
                    origin_text = self.mask_str_to_list(origin_text)
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

    
    def softmax(self, x, dim=None):
        x = x - x.max(axis=dim, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=dim, keepdims=True)
    
    def call_for_ascend(
        self,
        texts,
        only_mask=True,
        batch_size=32,
        return_topk=1
    ):
        """预测mask任务, 或者对所有字进行纠错, 升腾模式目前没有计算topk

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

        texts_for_bert_input =[ x.split(' ') for x in texts_for_bert_input]
        for start_idx in range(0, len(texts), batch_size):
            batch_outputs = []
            batch_texts = texts[start_idx:start_idx + batch_size]
            batch_texts_for_bert_input = texts_for_bert_input[start_idx:start_idx + batch_size]
            inputs = self.tokenizer(batch_texts_for_bert_input,
                                    return_tensors='np',
                                    )
      

            X1, X2, X3 = inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]
            preds = self.model.execute([X1, X2, X3])[0]

            for idx, origin_text in enumerate(batch_texts):

                if only_mask:
                    # MASK idx不能从tokenize后匹配, 因为tokenize后会过滤空格
                    pred_mask_idx = np.argwhere(inputs['input_ids'][idx] == self.tokenizer.mask_token_id).reshape(-1)

                    if len(pred_mask_idx) == 0:
                        # 如果没有mask
                        batch_outputs.append(
                            list(zip(origin_text, origin_text)))
                        continue
                    pred = preds[idx, pred_mask_idx, ...]
                else:
                    pred = preds[idx]

                pred_h = self.softmax(pred, dim=-1)

                pred_i = np.argmax(pred_h, axis=-1)
                
                pred_h = pred_h.tolist()
                mask_pred = self.tokenizer.convert_ids_to_tokens(pred_i)
                mask_pred = list([(tag, probs[idx])] for tag, idx, probs in zip(mask_pred, pred_i, pred_h)) 

                if only_mask:
                    origin_text = self.mask_str_to_list(origin_text)
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
    def call_for_ascend_wrapper(self,
        texts,
        batch_size=32,
        return_topk=1):

        if isinstance(texts, str):
            texts = [texts]

        # split batch for 1,2,4,8
        available_batch_size = [8,4,2,1]
        texts_num = len(texts)
        
        prepare_batch_size_li = []
        for availabel_bs in available_batch_size:
            chunk_num, remainder = texts_num // availabel_bs, texts_num % availabel_bs
            if chunk_num > 0:
                prepare_batch_size_li.extend([availabel_bs]*chunk_num)
            if remainder > 0:
                texts_num = remainder
            else:
                break
        
        # predict
        start_idx = 0
        outputs = []
        for bs in prepare_batch_size_li:
            text_chunk = texts[start_idx:start_idx+bs]
            start_idx += bs
            outputs.extend(self.call_for_ascend(text_chunk))

        return outputs
    @staticmethod
    def mask_str_to_list(text):
        """[summary]

        Args:
            text ([type]): '这 是一个[MASK]啊 ,你的[MASK]呢'

        Returns:
            [type]: ['这', ' ', '是', '一', '个', '[MASK]', '啊', ' ', ',', '你', '的', '[MASK]', '呢']
        """
        texts = text.split('[MASK]')
        if len(texts) < 2:
            return list(text)
        return list(reduce(lambda a, b: list(a) + ['[MASK]'] + list(b), texts))


if __name__ == '__main__':
    p = PredictorMlm(
        pretrained_model_dir='pretrained_model/chinese-roberta-wwm-ext')
    outputs = p(texts=['我将在公司挑选两名同[MASK]共同探讨这个方案]'],
                return_topk=2, only_mask=False)
    print(outputs)
