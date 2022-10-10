#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

import torch
#from logs import logger
from macbert_large_midu.gector.src.modeling.modeling_woe_s2e_electra import ModelingWoeS2eElectra
from macbert_large_midu.gector.src.tokenizer.bert_tokenizer import CustomBertTokenizer
from macbert_large_midu.gector.utils.data_helper import SPACE_SIGNS, include_cn, replace_punc_for_bert


class PredictorWoeSeq2Edit:
    def __init__(
        self,
        in_model_dir,
        ctc_label_vocab_dir='macbert_large_midu/gector/src/vocab',
        use_cuda=True,
        cuda_id=None,
    ):

        self.in_model_dir = in_model_dir
        self._id2dtag, self._dtag2id = self.load_label_dict(
            ctc_label_vocab_dir)
        self.tokenizer = CustomBertTokenizer.from_pretrained(in_model_dir)
        self.use_cuda = use_cuda
        self.cuda_id = cuda_id
        self.model = self.load_model()

    def load_model(self):

        model = ModelingWoeS2eElectra.from_pretrained(self.in_model_dir)
        model.eval()

        if self.use_cuda and torch.cuda.is_available():
            if self.cuda_id is not None:
                torch.cuda.set_device(self.cuda_id)
            model.cuda()
            model = model.half()  # 半精度
 #       logger.info('model loaded from: {}'.format(self.in_model_dir))

        return model

    def load_label_dict(self, ctc_label_vocab_dir):

        tag_file_name = 'disorder_tags.txt'
        dtag_fp = '{}/{}'.format(ctc_label_vocab_dir, tag_file_name)
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        print('d_tag num: {}, d_tag:{}'.format(len(id2dtag), id2dtag))

        return id2dtag, d_tag2id

    def id_list2ctag_list(self, id_list) -> list:

        return [self._id2ctag[i] for i in id_list]

    def id_list2dtag_list(self, id_list) -> list:

        return [self._id2dtag[i] for i in id_list]

    def predict(self, texts, batch_size=32, prob_threshold=0):
        """ seq2label
        Args:
            texts (list): 
            ignore_eng: 是否忽略英文检错
        Returns:
            List[tuple]: [ ('中', [ ('中', 0.9977), ('众', 0.0023)] ) ],
        """
        outputs = self._predict(texts, batch_size)
        outputs = [self.parse_predict_output(
            i, prob_threshold=prob_threshold) for i in outputs]
        return outputs

    @torch.no_grad()
    def _predict(
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
            c_preds, c_logits = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
            )
            for idx, length in enumerate(inputs['length'].tolist()):
                true_idx = range(1, length - 1)
                c_pred = c_preds[idx][1:-1] if isinstance(
                    c_preds, list) else c_preds[idx, true_idx, ...]
                c_logit = c_logits[idx, true_idx, ...]
                c_probs = torch.softmax(c_logit, dim=-1).tolist()
                c_pred_prob = [([self._id2dtag[pred_idx]], [probs[pred_idx]])
                               for pred_idx, probs in zip(c_pred, c_probs)]  # 先手动给个概率
                origin_text = batch_texts[idx]

                # 还原空格
                [[c_pred_prob.insert(i, (['$KEEP'], [1.0]))]
                 for i, v in enumerate(origin_text) if v in SPACE_SIGNS]

                c_pred_edits, c_pred_probs = list(zip(*c_pred_prob))
                outputs.append(list(
                    zip(origin_text, c_pred_edits, c_pred_probs)))

        return outputs

    @staticmethod
    def parse_predict_output(output, prob_threshold=0.3):

        src_text, pred_text = '', ''
        left_idxs, right_idxs, swap_prob_list = [], [], []
        for idx, (src_char, pred_c_char_list, pred_c_prob_list) in enumerate(output):
            src_text += src_char
            top1_edit = pred_c_char_list[0]
            top1_prob = pred_c_prob_list[0]

            if top1_edit == '$LEFT':
                left_idxs.append(idx)
                swap_prob_list.append(top1_prob)
            elif top1_edit == '$RIGHT':
                right_idxs.append(idx)
                swap_prob_list.append(top1_prob)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return src_text
        if left_idxs + right_idxs == list(range(left_idxs[0], right_idxs[-1]+1)):
            # 达到交换条件, 索引是连续等差数列
            pred_text = src_text[:left_idxs[0]] + src_text[right_idxs[0]:right_idxs[-1] +
                                                           1] + src_text[left_idxs[0]:left_idxs[-1]+1] + src_text[right_idxs[-1]+1:]

            if sum(swap_prob_list) / len(swap_prob_list) < prob_threshold:
                # filter by prob_threshold
                pred_text = src_text
            return pred_text
        return src_text


if __name__ == '__main__':
    #
    p = PredictorWoeSeq2Edit(
        in_model_dir='model/miduCTC_v3.7.0_csc3model/woe', use_cuda=True)

    # '请问董秘公司海外电商渠道销售都有那些平台...'
    r = p.predict(['今天的天气真错不！！', '今天的天气真错不！  ！'])

    print(r)
