#!/usr/bin/env python
# -*- coding: utf-8 -*-


from functools import reduce

import torch
# from logs import logger
from src.modeling.modeling_csc_realise import ModelingCscRealise
from src.tokenizer.bert_tokenizer import CustomBertTokenizer
from utils.data_helper import SPACE_SIGNS, include_cn, replace_punc_for_bert
from utils.pinyin_util import Pinyin2
from torch.cuda.amp import autocast
from utils.sound_shape_code.ssc import CharHelper

class PredictorCscRealise:
    def __init__(
        self,
        in_model_dir,
        max_len_char_pinyin=7,
        use_cuda=True,
        cuda_id=None,

    ):
        self.cuda_id = cuda_id
        self.in_model_dir = in_model_dir
        self.model = ModelingCscRealise.from_pretrained(
            in_model_dir)
        self.model.eval()
        # logger.info('model loaded from dir {}'.format(
        #     self.in_model_dir))
        self.use_cuda = use_cuda
        if self.use_cuda and torch.cuda.is_available():
            if self.cuda_id is not None:
                torch.cuda.set_device(self.cuda_id)
            self.model.cuda()
            self.model.half()
        
        self.max_len_char_pinyin = max_len_char_pinyin
        self.pho2_convertor = Pinyin2()
        self.tokenizer = CustomBertTokenizer.from_pretrained(in_model_dir)
        self.char_helper = CharHelper()

    
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
            inputs = self.tokenizer(batch_char_based_texts)
            input_chars = self.tokenizer.convert_ids_to_tokens(
                reduce(lambda a, b: a+b, inputs['input_ids']))
            phos_idx, phos_lens = self.pho2_convertor.convert(
                input_chars, self.max_len_char_pinyin)

            inputs['pho_idx'] = phos_idx
            inputs['pho_lens'] = torch.LongTensor(phos_lens)
            inputs['input_ids'] = torch.LongTensor(inputs['input_ids'])
            inputs['attention_mask'] = torch.LongTensor(
                inputs['attention_mask'])
            
            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                inputs['pho_idx'] = inputs['pho_idx'].cuda()
                inputs['pho_lens'] = inputs['pho_lens'].cuda()
            with autocast():
              d_preds, c_preds, pinyin_preds, loss = self.model(
                  input_ids=inputs['input_ids'],
                  attention_mask=inputs['attention_mask'],
                  pho_idx=inputs['pho_idx'],
                  pho_lens=inputs['pho_lens']
              )

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
    
    p = PredictorCscRealise(in_model_dir='model/csc_realise_ccl_finetune_2022Y08M09D18H/epoch5,step67,testepochf1_0.9938,devepochf1_0.993', use_cuda=True)
    r = p.predict(['上课的时候，一周是非常紧张的。'])
    print(r)