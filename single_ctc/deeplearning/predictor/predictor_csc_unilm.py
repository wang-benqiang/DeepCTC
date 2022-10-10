#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from functools import reduce

import torch
from src import logger
from src.deeplearning.modeling.modeling_unilm import ModelingUnilmCsc, ModelGpt, ModelingUnilmCscnobi, ModelingUnilmCscBiAtt
from src.utils.data_helper import SPACE_SIGNS, replace_punc_for_bert
from transformers.models.bert import BertForMaskedLM, BertTokenizer


class PredictorCscUnilm:
    def __init__(
        self,
        in_model_dir,
        model_type='bert',
        ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
        use_cuda=True,
        cuda_id=None,
    ):

        self.in_model_dir = in_model_dir

        self.model = ModelingUnilmCscBiAtt.from_pretrained(
            in_model_dir) 
  
        logger.info('model loaded from dir {}'.format(
            self.in_model_dir))
        self.use_cuda = use_cuda
        if self.use_cuda and torch.cuda.is_available():
            if cuda_id is not None:
                torch.cuda.set_device(cuda_id)
            self.model.cuda()
            self.model.half()
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(in_model_dir)

    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')

        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id

    def id_list2ctag_list(self, id_list) -> list:

        return [self._id2ctag[i] for i in id_list]

    def tokenize_inputs(self, texts, return_tensors=None):
        "预测tokenize, 按batch texts中最大的文本长度来pad, realise只需要input id, mask, length"

        cls_id, sep_id, pad_id, unk_id = self.tokenizer.vocab['[CLS]'], self.tokenizer.vocab[
            '[SEP]'], self.tokenizer.vocab['[PAD]'], self.tokenizer.vocab['[UNK]']
        input_ids, attention_mask, token_type_ids, length = [], [], [], []
        max_len = max([len(text) for text in texts]) + 2  # 注意+2

        for text in texts:
            true_input_id = [self.tokenizer.vocab.get(
                c, unk_id) for c in text][:max_len-2]
            pad_len = (max_len-len(true_input_id)-2)
            input_id = [cls_id] + true_input_id + [sep_id] + [pad_id] * pad_len

            a_mask = [1] * (len(true_input_id) + 2) + [0] * pad_len
            token_type_id = [0] * max_len
            input_ids.append(input_id)
            attention_mask.append(a_mask)
            token_type_ids.append(token_type_id)
            length.append(sum(a_mask))

        if return_tensors == 'pt':
            return {'input_ids': torch.LongTensor(input_ids),
                    'attention_mask': torch.LongTensor(attention_mask),
                    'token_type_ids': torch.LongTensor(token_type_ids),
                    'length': torch.LongTensor(length)}
        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'length': length}

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
            texts = [texts]
        else:
            texts = texts
        outputs = []
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx+batch_size]

            # batch_texts = [' ' + t for t in batch_texts] # 开头加一个占位符
            inputs = self.tokenize_inputs(batch_texts,
                                          return_tensors='pt')
            # inputs['input_ids'][..., 1] = 1  # 把 始 换成 [unused1]
            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

            final_hidden = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids']
            )
            # correct_h, l2r_h
            # final_hidden, l2r_h, r2l_h
            preds = final_hidden[0]
            # preds = torch.softmax(preds[:, 1:, :], dim=-1)  # 从cls后面开始
            preds = torch.softmax(preds[:, 0:, :], dim=-1)  
            recall_top_k_probs, recall_top_k_ids = preds.topk(
                k=return_topk, dim=-1, largest=True, sorted=True)
            recall_top_k_probs = recall_top_k_probs.tolist()
            recall_top_k_ids = recall_top_k_ids.tolist()
            recall_top_k_chars = [[self.tokenizer.convert_ids_to_tokens(
                char_level) for char_level in sent_level] for sent_level in recall_top_k_ids]
            # batch_texts = [ ['']+list(t)[1:] for t in batch_texts] # 占位符
            # batch_texts = [list(t) for t in batch_texts]  # 占位符
            batch_outputs = [list(zip(' '+ text + ' ', top_k_char, top_k_prob)) for text, top_k_char, top_k_prob in zip(
                batch_texts, recall_top_k_chars, recall_top_k_probs)]
            outputs.extend(batch_outputs)
        return outputs


if __name__ == '__main__':
    model_dir = 'model/csc_unilm_train_uni_atten_bi_2022Y06M17D17H/epoch3,ith_db0,step1,testf1_8_41%,devf1_9_35%,testloss790.2207,devloss55.7498,'
    # model_dir = 'model/csc_base_macbert_test_2022Y03M13D17H/epoch6,ith_db0, step1,testf1_0%,devf1_0%'
    model_dir2 = 'model/csc_unilm_train_uni_atten_bi2_2022Y06M17D18H/epoch3,ith_db0,step1,testf1_5_37%,devf1_6_39%,testloss933.738,devloss65.399,'
    # p = PredictorCscUnilm(in_model_dir=model_dir, use_cuda=False)
    p2 = PredictorCscUnilm(in_model_dir=model_dir2, use_cuda=False)

    # '请问董秘公司海外电商渠道销售都有那些平台...'中国的首都是北京。
    # r = p.predict(['始终与人民群众同甘共苦，为人民利益不懈奋斗，永葆共产党人政治本色','中国的首都是北京。'],  return_topk=5)
    r2 = p2.predict(['始终与人民群众同甘共苦，为人民利益不懈奋斗，永葆共产党人政治本色','中国的首都是北京。'],  return_topk=5)
    print(r2)
