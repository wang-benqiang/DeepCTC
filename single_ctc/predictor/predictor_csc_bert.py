#!/usr/bin/env python
# -*- coding: utf-8 -*-


from functools import reduce

import torch
#from logs import logger
from macbert_large_midu.gector.src.modeling.modeling_csc_bert import BertForMaskedLM
from macbert_large_midu.gector.src.tokenizer.bert_tokenizer import CustomBertTokenizer
from macbert_large_midu.gector.utils.data_helper import SPACE_SIGNS, include_cn, replace_punc_for_bert
from macbert_large_midu.gector.utils.pinyin_util import Pinyin2
from torch.cuda.amp import autocast


class PredictorCscBert:
    def __init__(
        self,
        in_model_dir,
        use_cuda=True,
        cuda_id=None,

    ):
        self.cuda_id = cuda_id
        self.in_model_dir = in_model_dir
        self.model = BertForMaskedLM.from_pretrained(
            in_model_dir)
        self.model.eval()
        #logger.info('model loaded from dir {}'.format(
        #    self.in_model_dir))
        self.use_cuda = use_cuda
        if self.use_cuda and torch.cuda.is_available():
            if self.cuda_id is not None:
                torch.cuda.set_device(self.cuda_id)
            self.model.cuda()
            self.model.half()
        
        self.tokenizer = CustomBertTokenizer.from_pretrained(in_model_dir)

    
    def predict(self, texts, return_topk=1, batch_size=32, prob_threshold=0):
        """ seq2label
        Args:
            texts (list): 
            ignore_eng: 是否忽略英文检错
        Returns:
            List[tuple]: [ ('中', [ ('中', 0.9977), ('众', 0.0023)] ) ],
        """
        outputs = self._predict(texts, return_topk, batch_size)
        outputs = [self.parse_predict_output(i, prob_threshold) for i in outputs]
        return outputs
    
    @torch.no_grad()
    def _predict(self, texts, return_topk=1, batch_size=32):
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
            inputs = self.tokenizer(batch_char_based_texts, return_tensors='pt')

            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
       
            with autocast():
              c_preds= self.model(
                  input_ids=inputs['input_ids'],
                  attention_mask=inputs['attention_mask'],
         
              )[0]

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
                origin_text = batch_texts[idx]
                # 还原空格

                [[pred.insert(i, [v]), pred_h.insert(i, [1.0])] for i, v in enumerate(
                    origin_text) if v in SPACE_SIGNS]
                outputs.append(list(zip(origin_text, pred, pred_h)))
        return outputs

    @staticmethod
    def parse_predict_output(output, prob_threshold=0):

        out_text = ''

        for (src_char, pred_c_char_list, pred_c_prob_list) in output:

            top1_char = pred_c_char_list[0]
            top1_char_prob = pred_c_prob_list[0]
            if top1_char != src_char and include_cn(top1_char) and include_cn(src_char) and top1_char_prob >= prob_threshold:
                out_text += top1_char
            else:
                out_text += src_char

        return out_text



if __name__ == '__main__':
    p = PredictorCscBert(in_model_dir='pretrained_model/macbert4csc-base-chinese', use_cuda=False)
    o = p._predict('如果那件事真话的话，我可以饶恕小新的陶器。', return_topk=5)
    print(o)
