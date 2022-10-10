#!/usr/bin/env python
# -*- coding: utf-8 -*-


from functools import reduce

import torch
#from logs import logger
from src.modeling.modeling_ctc_t5 import ModelingCtcT5, T5TokenizerFast
from utils.data_helper import SPACE_SIGNS, include_cn, replace_punc_for_bert
from utils.pinyin_util import Pinyin2
from torch.cuda.amp import autocast


class PredictorCtcT5:
    def __init__(
        self,
        in_model_dir,
        use_cuda=True,
        cuda_id=None,
    ):
        
        self.cuda_id = cuda_id
        self.in_model_dir = in_model_dir
        self.model = ModelingCtcT5.from_pretrained(
            in_model_dir)
        self.model.eval()
        logger.info('model loaded from dir {}'.format(
            self.in_model_dir))
        self.use_cuda = use_cuda
        if self.use_cuda and torch.cuda.is_available():
            if self.cuda_id is not None:
                torch.cuda.set_device(self.cuda_id)
            self.model.cuda()
            self.model.half()
        
        self.tokenizer = T5TokenizerFast.from_pretrained(in_model_dir)

    
    def predict(self, texts,  batch_size=32):
        """ seq2label
        Args:
            texts (list): 
            ignore_eng: 是否忽略英文检错
        Returns:
            List[tuple]: [ ('中', [ ('中', 0.9977), ('众', 0.0023)] ) ],
        """
        outputs = self._predict(texts, batch_size)
        return outputs
    
    @torch.no_grad()
    def _predict(self, texts,  batch_size=32):
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
        outputs_texts = []
        for start_idx in range(0, len(texts_inputs), batch_size):
            batch_texts = texts_inputs[start_idx:start_idx +
                                       batch_size]
            # 将一些bert词典中不存在的 中文符号 转换为 英文符号
            # 加空格让bert对英文分字
            batch_char_based_texts = [
                replace_punc_for_bert(text)
                for text in batch_texts
            ]
            inputs = self.tokenizer(batch_char_based_texts, 
                                    padding=True,
                                    truncation=True,
                                    return_tensors='pt')

            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
       
            with autocast():
              preds = self.model.generate(
                  input_ids=inputs['input_ids'],
              )
              batch_pred_texts = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
              outputs_texts.extend(batch_pred_texts)
              
        for idx, (src_text, pred_text) in enumerate(zip(texts , outputs_texts)):
            if len(src_text) != len(pred_text):
                _pred_text = src_text
            else:
                # remove punc
                _pred_text = ''.join([p if include_cn(s) and include_cn(p) else s for s, p in zip(src_text, pred_text)])
            outputs_texts[idx] = _pred_text
                    
        return outputs_texts

 



if __name__ == '__main__':
    p = PredictorCtcT5(in_model_dir='model/gec_t5_track2_finetune_cged_2022Y08M22D12H/epoch9,step222,testepochf1_0.17,devepochf1_0.5128', use_cuda=True)
    """
    ('我一直希望奶奶看到我长大变成什么人', '我一直希望奶奶看到我长大变成什么样的人')
    db_mng.get_src_trg(2)
    ('我们应该找时间可以好好地聊聊', '我们应该找时间好好地聊聊')
    db_mng.get_src_trg(1)
    ('我替你很高兴', '我替你高兴')
    db_mng.get_src_trg(2)
    ('我们应该找时间可以好好地聊聊', '我们应该找时间好好地聊聊')
    db_mng.get_src_trg(3)
    ('最近找到工作很难', '最近找工作很难')
    """
    o = p.predict(['1989、1990、1991年，他获得最优秀推销员赏。', '今天的天气天气很不错'])
    print(o)