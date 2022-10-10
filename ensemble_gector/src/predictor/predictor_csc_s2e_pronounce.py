#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import os

import numpy as np
import torch
# from logs import logger
from src.modeling.modeling_csc_s2e_pronounce import ModelingCscS2ePronounce
from src.tokenizer.bert_tokenizer import CustomBertTokenizer
from utils.data_helper import SPACE_SIGNS, include_cn, replace_punc_for_bert


class PredictorCscS2ePronounce:
    def __init__(
        self,
        in_model_dir,
        ctc_label_vocab_dir='src/vocab',
        use_cuda=True,
        cuda_id=None,
    ):

        self.in_model_dir = in_model_dir
        self.id2dtag, self.dtag2id, self.id2ctag, self.ctag2id, self.char2shengmu_dict, self.char2yunmu_dict = self.load_label_dict(
            ctc_label_vocab_dir)
        self.tokenizer = CustomBertTokenizer.from_pretrained(in_model_dir)
        self.use_cuda = use_cuda
        self.cuda_id = cuda_id
        self.model = self.load_model()

    def load_model(self):

        model = ModelingCscS2ePronounce.from_pretrained(self.in_model_dir)
        model.eval()

        if self.use_cuda and torch.cuda.is_available():
            if self.cuda_id is not None:
                torch.cuda.set_device(self.cuda_id)
            model.cuda()

            model = model.half()  # 半精度
        logger.info('model loaded from: {}'.format(self.in_model_dir))

        return model

    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')
        char2shengmu_fp = os.path.join(ctc_label_vocab_dir, 'char_shengmu.json')
        char2yunmu_fp = os.path.join(ctc_label_vocab_dir, 'char_yunmu.json')
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}

        # char to pinyin code
        char2shengmu_dict = json.load(
            open(char2shengmu_fp, 'r', encoding='utf8'))
        char2yunmu_dict = json.load(
            open(char2yunmu_fp, 'r', encoding='utf8'))

        char2shengmu_dict['[UNK]']
        # pinyin code to id
        pycode2id = {
            str(v): idx for idx, v in enumerate(list(range(0, 10)) + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        }
        length = len(pycode2id)
        pycode2id['[PAD]'] = length
        pycode2id['[UNK]'] = length+1

        char2shengmu_dict = {char: pycode2id.get(
            code, pycode2id['[UNK]']) for char, code in char2shengmu_dict.items()}
        char2yunmu_dict = {char: pycode2id.get(
            code, pycode2id['[UNK]']) for char, code in char2yunmu_dict.items()}

        logger.info('d_tag num: {}, detect_labels:{}'.format(
            len(id2dtag), d_tag2id))

        return id2dtag, d_tag2id, id2ctag, c_tag2id, char2shengmu_dict, char2yunmu_dict

    def id_list2ctag_list(self, id_list) -> list:

        return [self.id2ctag[i] for i in id_list]

    def id_list2dtag_list(self, id_list) -> list:

        return [self.id2dtag[i] for i in id_list]

    def get_pinyin_info(self, texts, max_seq_len, return_tensors='pt'):

        input_shengmu, input_yunmu = [], []

        if isinstance(texts, str):
            texts = texts

        for text in texts:
            ignore_loss_seq_len = max_seq_len-(len(text)+1)  # sep and pad

            text_shengmu = [self.char2shengmu_dict['[PAD]']] + [self.char2shengmu_dict.get(char, self.char2shengmu_dict['[UNK]']) if include_cn(
                char) else self.char2shengmu_dict['[UNK]'] for char in text] + [self.char2shengmu_dict['[PAD]']] * ignore_loss_seq_len

            text_yunmu = [self.char2yunmu_dict['[PAD]']] + [self.char2yunmu_dict.get(char, self.char2yunmu_dict['[UNK]']) if include_cn(
                char) else self.char2yunmu_dict['[UNK]'] for char in text] + [self.char2yunmu_dict['[PAD]']] * ignore_loss_seq_len

            input_shengmu.append(text_shengmu)
            input_yunmu.append(text_yunmu)

        if return_tensors == 'pt':
            input_shengmu = torch.LongTensor(input_shengmu)
            input_yunmu = torch.LongTensor(input_yunmu)
        return input_shengmu, input_yunmu

    def predict(self, texts, return_topk=1, batch_size=32, prob_threshold=0):
        """ seq2label
        Args:
            texts (list): 
            ignore_eng: 是否忽略英文检错
        Returns:
            List[tuple]: [ ('中', [ ('中', 0.9977), ('众', 0.0023)] ) ],
        """
        outputs = self._predict(texts, return_topk, batch_size)
        outputs = [self.parse_predict_output(i, prob_threshold=prob_threshold) for i in outputs]
        return outputs
    
    @torch.no_grad()
    def _predict(
        self,
        texts,
        batch_size=32,
        return_topk=1,
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
            max_seq_len = inputs['input_ids'].shape[-1]
            inputs['input_shengmu'], inputs['input_yunmu'] = self.get_pinyin_info(
                batch_char_based_texts, max_seq_len, return_tensors='pt')

            if self.use_cuda and torch.cuda.is_available():
                for k, v in inputs.items():
                    inputs[k] = v.cuda()
            model_preds = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
                input_shengmu=inputs['input_shengmu'],
                input_yunmu=inputs['input_yunmu'],
            )

            batch_c_preds, batch_d_preds = model_preds[0], model_preds[1]

            for idx, length in enumerate(inputs['length'].tolist()):
                true_idx = range(1, length - 1)
                # true_idx = range(0, length - 0)
                c_preds = batch_c_preds[idx, true_idx, ...]
                d_preds = batch_d_preds[idx, true_idx, ...]

                c_pred_prob = torch.softmax(c_preds, dim=-1)
                d_pred_prob = torch.softmax(d_preds, dim=-1)
                # care detect output
                c_pred_prob[..., 1] -= d_pred_prob[..., 1]
                c_pred_prob, c_pred_idx = c_pred_prob.topk(k=return_topk,
                                                           dim=-1,
                                                           largest=True,
                                                           sorted=True)  # logit, idx

                # d_pred_prob, d_pred_idx = d_pred_prob.topk(k=2,
                #                            dim=-1,
                #                            largest=True,
                #                            sorted=True)  # logit, idx

                c_pred_prob = c_pred_prob.tolist()
                # d_pred_prob = d_pred_prob.tolist()

                c_pred_tag = [self.id_list2ctag_list(x) for x in c_pred_idx]
                # d_pred_tag = [self.id_list2dtag_list(x) for x in d_pred_idx]

           
                # d_pred_tag_prob = list(list(zip(*ele)) for ele in zip(d_pred_tag, d_pred_prob))
                origin_text = batch_texts[idx]
                # 还原空格

                [[c_pred_tag.insert(i, [v]), c_pred_prob.insert(i, [1])] for i, v in enumerate(
                    origin_text) if v in SPACE_SIGNS]

                origin_char_list = list(origin_text)
                outputs.append(list(zip(origin_char_list, c_pred_tag, c_pred_prob)))
                # outputs.append(list(zip(origin_char_list, c_pred_tag_prob, d_pred_tag_prob)))

        return outputs

    
    
    @staticmethod
    def parse_predict_output(output, prob_threshold=0):
        
        out_text = ''
        
        for (src_char, pred_c_char_list, pred_c_prob_list) in output:
            
            top1_char = pred_c_char_list[0].split('_')[-1]
            top1_char_prob = pred_c_prob_list[0]
            if top1_char != '$KEEP' and include_cn(top1_char) and include_cn(src_char) and top1_char_prob >= prob_threshold:
                out_text += top1_char
                
            else:
                out_text += src_char
            
        return out_text


if __name__ == '__main__':
    in_model_dir = 'model/miduCTC_v3.8.0_csc2model/csc3'
    p = PredictorCscS2ePronounce(in_model_dir, use_cuda=True)

    # '请问董秘公司海外电商渠道销售都有那些平台...'
    r = p.predict(['她现在想旅游计划，在网路上找好玩、好吃的东西。', '每次访客搬出去，我们不得不装修我们的房子。'],  return_topk=3)
    print(r)
