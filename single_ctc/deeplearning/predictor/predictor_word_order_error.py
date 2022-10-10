#!/usr/bin/env python
# -*- coding: utf-8 -*-


import re
from functools import reduce
import numpy as np
from configs.ctc_conf import CtcConf
import torch
from src import logger
from src.deeplearning.modeling.modeling_word_order_error import WordOrderErrorModel
from src.utils.data_helper import (SPACE_SIGNS, include_cn,
                                   replace_punc_for_bert)
from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
from transformers.models.electra import ElectraTokenizer

"乱序"


class PredictorWordOrderError:
    def __init__(
        self,
        pretrained_model_dir,
        ctc_label_vocab_dir,
        pretrained_model_type='electra',
        use_cuda=True,
        cuda_id=None,
        onnx_mode=False
    ):

       
        self.pretrained_model_dir = pretrained_model_dir
        self.use_cuda = use_cuda

        self.pretrained_model_type = pretrained_model_type
        self.eng_digit_re = re.compile(r'[a-zA-Z]|[0-9]')

    
        self.tokenizer = CustomBertTokenizer.from_pretrained(
            pretrained_model_dir)

        self.id2dtag, self.d_tag2id = self.load_label_dict(
            ctc_label_vocab_dir)
        self.cuda_id = cuda_id
        self.onnx_mode = onnx_mode
        self.model = self.load_model()


   

    
    def load_model(self):
        
        if CtcConf.ascend_mode:
            from src.corrector.corrector import AclLiteModel
            model = AclLiteModel(CtcConf.woe_detect_model_fp_ascend_mode)
            logger.info('model loaded from: {}'.format(CtcConf.woe_detect_model_fp_ascend_mode))
        else:
            
            model = WordOrderErrorModel.from_pretrained(self.pretrained_model_dir)
            model.eval()
     
            if self.use_cuda and torch.cuda.is_available():
                if self.cuda_id is not None:
                    torch.cuda.set_device(self.cuda_id)
                model.cuda()

                if not self.onnx_mode:
                    model = model.half()  # 半精度
            logger.info('model loaded from: {}'.format(self.pretrained_model_dir))

        return model
    def load_label_dict(self, ctc_label_vocab_dir):

       
        tag_file_name = 'disorder_tags.txt'
        dtag_fp = '{}/{}'.format(ctc_label_vocab_dir, tag_file_name)
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        print('d_tag num: {}, d_tag:{}'.format(len(id2dtag), id2dtag))

        return id2dtag, d_tag2id


    @torch.no_grad()
    def predict(
        self,
        texts,
        batch_size=32,
        only_care_cn=True
    ):
        """ seq2label

        Args:
            texts (list): 
            only_mask (bool, optional): 只对句子中mask的部分进行预测. Defaults to False.
            ignore_idx_list: [[12], [13,14]]
        Returns:
            List[tuple]: [('你','$KEEP'), ('好', '$DELETE')],
            or
            List[str], 
                        [('你','$KEEP'), ('好', '$DELETE')],
                    ]
        """
        
        if CtcConf.ascend_mode:
            return self.predict_on_ascend_wrapper(texts, batch_size,)
        outputs = []
        if isinstance(texts, str):
            texts = [texts]

        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]
            # 将一些bert词典中不存在的 中文符号 转换为 英文符号
            # 加空格让bert对英文分字
            batch_char_based_texts = [
                replace_punc_for_bert(text)
                for text in batch_texts
            ]

            inputs = self.tokenizer(
                batch_char_based_texts, return_tensors='pt')

             

            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

            preds, sequence_outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
            )
            for idx, length in enumerate(inputs['length'].tolist()):
                true_idx = range(1, length-1)
                
                pred = preds[idx][1:-1] if isinstance(preds, list) else preds[idx, true_idx, ...] 
                sequence_output = sequence_outputs[idx, true_idx, ...]
                if self.model.config.append_crf:
                    pred_prob = torch.softmax(sequence_output, dim=-1).tolist()
                    pred = [(self.id2dtag[p], probs[p]) for p, probs in zip(pred, pred_prob)]  #先手动给个概率
                    
                else:
                    
                    pred_prob = torch.softmax(pred, dim=-1).tolist()
                    pred_idx = torch.argmax(pred, dim=-1).tolist()
                    pred = [(self.id2dtag[i], p[i]) for i, p in zip(pred_idx, pred_prob)]
               
                origin_text_list = list(batch_texts[idx])
                # 把原来文本中的空格还原到pred中，
               
                [
                    pred.insert(i, '$KEEP')
                    for i, v in enumerate(origin_text_list) if v in SPACE_SIGNS
                ]
               
                output = list(zip(origin_text_list, pred))
                if only_care_cn:
                    output = list(
                        map(
                            lambda char_dtag: (
                                char_dtag[0], ('$KEEP', 'no_cn'))
                            if not include_cn(char_dtag[0]) and char_dtag[0]!='[CLS]' else char_dtag,
                            output))
                outputs.append(output)
        return outputs

  
    def softmax(self, x, dim=None):
        x = x - x.max(axis=dim, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=dim, keepdims=True)
    
    def predict_on_ascend(
        self,
        texts,
        batch_size=32,
        only_care_cn=True
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
      
            # 将一些bert词典中不存在的 中文符号 转换为 英文符号
            # 加空格让bert对英文分字
            batch_char_based_texts = [
                replace_punc_for_bert(text)
                for text in batch_texts
            ]
            inputs = self.tokenizer(
                batch_char_based_texts, max_len=128,return_tensors='np')

            X1, X2, X3 = inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]
            c_preds = self.model.execute([X1, X2, X3])[0]
            c_preds = torch.from_numpy(c_preds).float()
            for idx, length in enumerate(inputs['length'].tolist()):
                true_idx = range(1, length - 1)
                pred = c_preds[idx, true_idx, ...]
                pred_h = torch.softmax(pred, dim=-1)
                # "softmax_lastdim_kernel_impl" not implemented for 'Half'
                pred_i = torch.argmax(pred_h, axis=-1).tolist()
                pred_h = pred_h.tolist()
                
                pred = [self.id2dtag[int(i)] for i in pred_i]
                pred = list((tag, probs[idx]) for tag, idx, probs in zip(pred, pred_i, pred_h)) #暂时只取第一个
                origin_text = batch_texts[idx]
                # 还原空格

                [pred.insert(i, [(v, 1.0)]) for i, v in enumerate(
                    origin_text) if v in SPACE_SIGNS]
                
       
                output = list(zip(origin_text, pred))
                if only_care_cn:
                    output = list(
                        map(
                            lambda char_dtag: (
                                char_dtag[0], ('$KEEP', 'no_cn'))
                            if not include_cn(char_dtag[0]) and char_dtag[0]!='[CLS]' else char_dtag,
                            output))
                outputs.append(output)
                
        return outputs
    
    def predict_on_ascend_wrapper(self,
        texts,
        batch_size=32,
        return_topk=1):

        if isinstance(texts, str):
            texts = [texts]

        # split batch for 1,2,4,8
        available_batch_size = [8,4,2,1]
        texts_num = len(texts)
        
        prepare_batch_size_li = []
        for availabel_bs in available_batch_size:
            chunk_num, remainder = texts_num // availabel_bs, texts_num % availabel_bs
            if chunk_num > 0:
                prepare_batch_size_li.extend([availabel_bs]*chunk_num)
            if remainder > 0:
                texts_num = remainder
            else:
                break
        
        # predict
        start_idx = 0
        outputs = []
        for bs in prepare_batch_size_li:
            text_chunk = texts[start_idx:start_idx+bs]
            start_idx += bs
            outputs.extend(self.predict_on_ascend(text_chunk))

        return outputs