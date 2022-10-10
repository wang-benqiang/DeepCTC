#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from functools import reduce
from tkinter.tix import Tree

import numpy as np
import torch
# from logs import logger
from src.modeling.modeling_csc_mlm import ModelMlm
from src.tokenizer.bert_tokenizer import CustomBertTokenizer
from utils.data_helper import SPACE_SIGNS, include_cn, replace_punc_for_bert
from utils.sound_shape_code.ssc import CharHelper

class PredictorCscMlm:
    def __init__(
            self,
            in_model_dir,
            use_cuda=True,
            cuda_id=0
    ):

        self.in_model_dir = in_model_dir
        self.use_cuda = use_cuda
        self.cuda_id = cuda_id

        self.eng_digit_re = re.compile(r'[a-zA-Z]|[0-9]')

        self.tokenizer = CustomBertTokenizer.from_pretrained(
            in_model_dir)
        self.model = self.load_model()
        self.char_helper = CharHelper()

    def load_model(self):

        model = ModelMlm.from_pretrained(self.in_model_dir)
        model.eval()
        if self.use_cuda and torch.cuda.is_available():
            if self.cuda_id is not None:
                torch.cuda.set_device(self.cuda_id)
                model.cuda()
                model.half()
        logger.info('model loaded: {}'.format(self.in_model_dir))
        return model

    
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
    def _predict(self, texts, return_topk=3, batch_size=32):
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
            inputs = self.tokenizer(batch_texts,
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

    
    
    
    def parse_predict_output(self, output, prob_threshold=0.88, similarity_2char_threshold=0.3):

        out_text = ''
        error_dict = {}
        for idx, (src_char, pred_c_char_list, pred_c_prob_list) in enumerate(output):

            top1_char = pred_c_char_list[0]
            #0。88
            if src_char != top1_char and pred_c_prob_list[0]> 0.89 and include_cn(src_char) and include_cn(top1_char) and self.char_helper.compute_sound_similarity_2char(src_char, top1_char) == 1:
                # 如果top1 拼音一样选择top1
                 error_dict[idx] = [top1_char, 1]
                 continue
                
            elif src_char not in pred_c_char_list and include_cn(src_char):
                temp_pred_char = src_char
                
                for pred_char, pred_prob in zip(pred_c_char_list, pred_c_prob_list):
                    if include_cn(pred_char) and pred_prob> prob_threshold and self.char_helper.compute_sound_similarity_2char(src_char, pred_char)>=similarity_2char_threshold:
                        temp_pred_char = pred_char
                        break
                error_dict[idx] = [temp_pred_char, pred_prob]
                
                
                
        
        src_text_li = [i[0] for i in output]
        if error_dict:
            sorted_error_by_prob = sorted(error_dict.items(), key=lambda x : x[1][1], reverse=True)
            idx, pred_char_with_prob = sorted_error_by_prob[0]
            src_text_li[idx] = pred_char_with_prob[0]
            
        return ''.join(src_text_li)

if __name__ == '__main__':

    mlm_csc_model_dir = 'pretrained_model/chinese-roberta-wwm-ext'
    # mlm_model_dir = 'model/mlm_roberta_base_pretrained_2022Y03M22D12H/epoch1,ith_db1, step98274,testf1_54_05%,devf1_66_36%'
    p = PredictorCscMlm(
        in_model_dir=mlm_csc_model_dir, use_cuda=True)
    
    outputs = p._predict(
        texts=['小明那么早起床是因为他的学校在很远的地方。'], return_topk=6,)
    print(outputs)
    
