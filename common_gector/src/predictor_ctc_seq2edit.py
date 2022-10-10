#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

import torch
# from logs import logger
from src.modeling_ctc_s2e_bert import ModelingCtcS2eBert
from src.bert_tokenizer import CustomBertTokenizer
from src.data_helper import SPACE_SIGNS, include_cn, replace_punc_for_bert


class PredictorCtcSeq2Edit:
    "错多少乱序"
    def __init__(
        self,
        in_model_dir,
        ctc_label_vocab_dir='../ensemble_gector/src/vocab',
        use_cuda=True,
        cuda_id=None,
    ):

        self.in_model_dir = in_model_dir
        self._id2dtag, self._dtag2id, self._id2ctag, self._ctag2id = self.load_label_dict(
            ctc_label_vocab_dir)
        self.tokenizer = CustomBertTokenizer.from_pretrained(in_model_dir)
        self.use_cuda = use_cuda
        self.cuda_id = cuda_id
        self.model = self.load_model()

    def load_model(self):

        model = ModelingCtcS2eBert.from_pretrained(self.in_model_dir)
        model.eval()

        if self.use_cuda and torch.cuda.is_available():
            if self.cuda_id is not None:
                torch.cuda.set_device(self.cuda_id)
            model.cuda()
            model = model.half()  # 半精度
        # logger.info('model loaded from: {}'.format(self.in_model_dir))

        return model

    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')

        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        # logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id

    def id_list2ctag_list(self, id_list) -> list:

        return [self._id2ctag[i] for i in id_list]

    def id_list2dtag_list(self, id_list) -> list:

        return [self._id2dtag[i] for i in id_list]

    
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

            if self.use_cuda and torch.cuda.is_available():
                for k, v in inputs.items():
                    inputs[k] = v.cuda()
            c_preds, d_preds, loss = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
            )

            for idx, length in enumerate(inputs['length'].tolist()):
                # true_idx = range(1, length - 1)
                # ctc任务开头可能少字
                true_idx = range(0, length - 1)
                c_pred = c_preds[idx, true_idx, ...]
                d_pred = d_preds[idx, true_idx, ...]
                c_pred_prob = torch.softmax(c_pred, dim=-1)
                d_pred_prob = torch.softmax(d_pred, dim=-1)
                
                "c_prob - d_prob"
                c_pred_prob[:, 1] -= d_pred_prob[:, 1]  
                
                c_pred_prob, c_pred_id = c_pred_prob.topk(k=return_topk,
                                                          dim=-1,
                                                          largest=True,
                                                          sorted=True)  # logit, idx
                d_pred_prob, d_pred_id = d_pred_prob.topk(k=2,
                                                          dim=-1,
                                                          largest=True,
                                                          sorted=True)  # logit, idx
                c_pred_prob = c_pred_prob.tolist()
                d_pred_prob = d_pred_prob.tolist()
                c_pred_chars = [self.id_list2ctag_list(x) for x in c_pred_id]
                d_pred_chars = [self.id_list2dtag_list(x) for x in d_pred_id]
                origin_char_list = ['']+list(batch_texts[idx])
                # 还原空格
                [[c_pred_chars.insert(i, ['$KEEP']), c_pred_prob.insert(i, [1.0])]
                 for i, v in enumerate(origin_char_list) if v in SPACE_SIGNS]
                # 还原空格
                [[d_pred_chars.insert(i, ['$RIGHT']), d_pred_prob.insert(
                    i, [1.0])] for i, v in enumerate(origin_char_list) if v in SPACE_SIGNS]
           
                outputs.append(list(
                    zip(origin_char_list, c_pred_chars, c_pred_prob, d_pred_chars, d_pred_prob)))

        return outputs


    @staticmethod
    def parse_predict_output(output, prob_threshold=0):
        
        out_text = ''
        src_text = []
        delete_idxs, replace_idx_dict, left_idxs, right_idxs = [], {}, [], []
        
        for idx, (src_char, pred_c_char_list, pred_c_prob_list, pred_d_tag_list, pred_d_pron_list) in enumerate(output):
            src_text.append(src_char)
            top1_char = pred_c_char_list[0]
            top1_char_prob = pred_c_prob_list[0]
            if top1_char == '$DELETE' and include_cn(src_char) and top1_char_prob >= prob_threshold:
                delete_idxs.append(idx)
            elif 'APPEND_' in top1_char and top1_char_prob >= prob_threshold:
                replace_idx_dict[idx] = src_char + top1_char.split('_')[-1]
            elif 'REPLACE_' in top1_char and top1_char_prob >= prob_threshold:
                replace_idx_dict[idx] = top1_char.split('_')[-1]
            elif '$LEFT' in top1_char and top1_char_prob >= prob_threshold:
                left_idxs.append(idx)
            elif '$RIGHT' in top1_char and top1_char_prob >= prob_threshold:
                right_idxs.append(idx)
        
        for delete_idx in delete_idxs:
            src_text[delete_idx] = ''
        for replace_idx, chars in replace_idx_dict.items():
            src_text[replace_idx] = chars
        
        if left_idxs and right_idxs and left_idxs + right_idxs == list(range(left_idxs[0], right_idxs[-1]+1)):
            src_text = src_text[:left_idxs[0]] + src_text[right_idxs[0]:right_idxs[-1] +
                                                           1] + src_text[left_idxs[0]:left_idxs[-1]+1] + src_text[right_idxs[-1]+1:]
        out_text = ''.join(src_text)
        return out_text
    
    
if __name__ == '__main__':
    # 
    p = PredictorCtcSeq2Edit(
        in_model_dir='model/gector/finetune_final_2022Y08M25D15H/normal_model/epoch10,step503,testepochf1_0.2939,devepochf1_0.5674',
        use_cuda=True)

    # '请问董秘公司海外电商渠道销售都有那些平台...'
    r = p.predict(['上课的时候，一周是非常紧张的。'],  return_topk=5)
    
    print(r)
