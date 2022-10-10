#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import torch
# from logs import logger
from src.modeling.modeling_gpt import ModelingGpt2
from src.tokenizer.bert_tokenizer import CustomBertTokenizer
from utils.data_helper import SPACE_SIGNS, include_cn, replace_punc_for_bert
from sklearn.metrics import classification_report


"csc task: detect sentence, token"


class PredictorGpt:
    def __init__(
        self,
        in_model_dir,
        ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
        use_cuda=True,
        cuda_id=None,
        onnx_mode=False,
        batch_size=48
    ):

        self.in_model_dir = in_model_dir
        self.tokenizer = CustomBertTokenizer.from_pretrained(in_model_dir)
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.cuda_id = cuda_id
        self.model = self.load_model()

    def load_model(self):

        model = ModelingGpt2.from_pretrained(self.in_model_dir)
        model.init_criterion()
        model.eval()
        if self.use_cuda and torch.cuda.is_available():
            if self.cuda_id is not None:
                torch.cuda.set_device(self.cuda_id)
            model.cuda()

         
            model = model.half()  # 半精度
        logger.info('model loaded from: {}'.format(self.in_model_dir))

        return model

    def id_list2tag_list(self, id_list) -> list:

        return [self.tokenizer.convert_ids_to_tokens(i) for i in id_list.tolist()]
    
    # def predict(self, texts, return_topk=1, batch_size=32):
    #     """ seq2label
    #     Args:
    #         texts (list): 
    #         ignore_eng: 是否忽略英文检错
    #     Returns:
    #         List[tuple]: [ ('中', [ ('中', 0.9977), ('众', 0.0023)] ) ],
    #     """
    #     outputs = self._predict(texts, return_topk, batch_size)
    #     outputs = [self.parse_predict_output(i) for i in outputs]
    #     return outputs
    
    @torch.no_grad()
    def predict(
        self,
        texts,
        return_topk=1,
    ):
        """ seq2label

        Args:
            texts (list):
            ignore_idx_list: [[12], [13,14]]
        Returns:
            List[tuple]: [('你','$KEEP'), ('好', '$DELETE')],
        """

        token_outputs, lm_scores = [], []

        if isinstance(texts, str):
            texts = [texts]

        for start_idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[start_idx:start_idx + self.batch_size]

            # 将一些bert词典中不存在的 中文符号 转换为 英文符号
            # 加空格让bert对英文分字

            batch_char_based_texts = [
                replace_punc_for_bert(text)
                for text in batch_texts
            ]

            inputs = self.tokenizer(
                batch_char_based_texts, return_tensors='pt')

            if self.use_cuda and torch.cuda.is_available():
                for k, v in inputs.items():
                    inputs[k] = v.cuda()
            
            token_logits, token_losses = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                is_training = False
            )
            
            token_losses = token_losses.view(token_logits.shape[0], -1)
            batch_lm_scores = -torch.sum(token_losses, dim=-1) / (inputs['length'] - 1)
            batch_lm_scores = batch_lm_scores.tolist()
            lm_scores.extend(batch_lm_scores)
            for idx, length in enumerate(inputs['length'].tolist()):

                # token level
                true_idx = range(0, length - 2)

                pred_token_logits = token_logits[idx, true_idx, ...]
                pred_token_loss = token_losses[idx, true_idx, ...]

                pred_token_prob = torch.softmax(pred_token_logits, dim=-1)

                pred_token_prob, pred_token_top_idx = pred_token_prob.topk(k=return_topk,
                                                                           dim=-1,
                                                                           largest=True,
                                                                           sorted=True)  # logit, idx
                pred_prob_list = pred_token_prob.tolist()
                pred_token_loss = pred_token_loss.tolist()
                pred_char_list = [self.id_list2tag_list(
                    x) for x in pred_token_top_idx]

                origin_text = batch_texts[idx]
                # 还原空格

                [[pred_char_list.insert(i, [v]), pred_prob_list.insert(i, [1.0]), pred_token_loss.insert(i, 0)] for i, v in enumerate(
                    origin_text) if v in SPACE_SIGNS]

                # origin_char_list = [''] + list(origin_text)
                origin_char_list = list(origin_text)
                token_outputs.append(
                    list(zip(origin_char_list, pred_char_list, pred_prob_list, pred_token_loss)))

        return token_outputs, lm_scores




if __name__ == '__main__':

    # p = PredictorGpt(
    #     in_model_dir='model/gpt2_pretrain_2022Y07M21D19H/epoch2,step23079,testepochloss_118.17,devepochloss_17.89', use_cuda=False)
    
    
    p = PredictorGpt(
        in_model_dir='model/gpt2_pretrain_lmdb_0207_1_2022Y07M26D17H/epoch1,step137822,testepochloss_115.86,devepochloss_16.09', use_cuda=False)

    # 即时做得不容易，我也想试试看。
    # 我年龄不少，睡眠不足的话明天也会受到影响的。
    # 在那里有一位我中国朋友的先辈。
    # 我要再开始用工学习汉语！
    # 我在我第一份工作里拼命得工作差不多半年多了。
    # 我反覆五次问她的名字。
    # 她是奥林匹克径赛的选手。
    
    
    
    token_outputs, lm_scores = p.predict(['夏天的时候很多客人去哪里。', ' 夏天的时候很多客人去那里。'],  return_topk=5)
    print(token_outputs)
