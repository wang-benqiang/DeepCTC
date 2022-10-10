#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

import numpy as np
import torch
from configs.ctc_conf import CtcConf
from src import logger
from src.deeplearning.modeling.modeling_tc import ModelingBertForTC
from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
from src.utils.data_helper import SPACE_SIGNS, replace_punc_for_bert

"text classification"


class PredictorTc:
    def __init__(
        self,
        in_model_dir,
        ctc_label_vocab_dir='src/deeplearning/ctc_vocab/tc_tags.txt',
        activate_prob=0.5,
        use_cuda=True,
        cuda_id=None,
    ):

        self.in_model_dir = in_model_dir
        self.id2tag, self.tag2id = self.load_label_dict(
            ctc_label_vocab_dir)
        self.tokenizer = CustomBertTokenizer.from_pretrained(in_model_dir)
        self.use_cuda = use_cuda
        self.cuda_id = cuda_id
        self.activate_prob = activate_prob
        self.model = self.load_model()

    def load_model(self):

        model = ModelingBertForTC.from_pretrained(self.in_model_dir)
        model.eval()

        if self.use_cuda and torch.cuda.is_available():
            if self.cuda_id is not None:
                torch.cuda.set_device(self.cuda_id)
            model.cuda()

            model = model.half()  # 半精度
        logger.info('model loaded from: {}'.format(self.in_model_dir))

        return model

    def load_label_dict(self, tc_vocab_fp):

        id2tag = [line.strip() for line in open(tc_vocab_fp, encoding='utf8')]
        tag2id = {v: i for i, v in enumerate(id2tag)}
        logger.info('tag num: {}, tags:{}'.format(len(id2tag), tag2id))
        return id2tag, tag2id

    def convert_prob2tag_list(self, prob_list) -> list:

        return [self.id2tag[idx] for idx, prob in enumerate(prob_list) if prob >= self.activate_prob]

    @torch.no_grad()
    def predict(
        self,
        texts,
        batch_size=32,

    ):
        """ seq2label

        Args:
            texts (list):
            ignore_idx_list: [[12], [13,14]]
        Returns:
            List[tuple]: [('你','$KEEP'), ('好', '$DELETE')],
        """

        outputs = []
        if isinstance(texts, str):
            texts = [texts]

        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]

            batch_char_based_texts = [
                replace_punc_for_bert(text)
                for text in batch_texts
            ]
            inputs = self.tokenizer(
                batch_char_based_texts, return_tensors='pt')

            if self.use_cuda and torch.cuda.is_available():
                for k, v in inputs.items():
                    inputs[k] = v.cuda()
            preds = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
            )[0]
          

            pred_h = torch.sigmoid(preds)

            pred_h = pred_h.tolist()

            pred_tag = [self.convert_prob2tag_list(x) for x in pred_h]

            activate_pred_h = [[i for i in p if i >=
                                self.activate_prob] for p in pred_h]

            outputs.append(list(zip(batch_texts, pred_tag, activate_pred_h)))

        return outputs


if __name__ == '__main__':
    from configs.ctc_conf import CtcConf

    p = PredictorTc(
        in_model_dir='model/tc_train_law_wiki_ancient_2022Y07M21D18H/epoch3,step1,testf1_None,devf1_70_4%', use_cuda=True)

    # '请问董秘公司海外电商渠道销售都有那些平台...'
    r = p.predict(['今老母已丧，抱恨终天。', '当晚二十一点，发生一起抢劫事件'])
    print(r)
