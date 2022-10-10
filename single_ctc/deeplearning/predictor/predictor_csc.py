#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
from configs.ctc_conf import CtcConf
import torch
from src import logger
from src.deeplearning.modeling.modeling_ctc import ModelingCTC
from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
from src.utils.data_helper import SPACE_SIGNS, replace_punc_for_bert



class PredictorCsc:
    def __init__(
        self,
        in_model_dir,
        ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
        use_cuda=True,
        cuda_id=None,
        onnx_mode=False
    ):

        self.in_model_dir = in_model_dir
        self._id2dtag, self._dtag2id, self._id2ctag, self._ctag2id = self.load_label_dict(
            ctc_label_vocab_dir)
        self.tokenizer = CustomBertTokenizer.from_pretrained(in_model_dir)
        self.use_cuda = use_cuda
        self.onnx_mode = onnx_mode
        self.cuda_id = cuda_id
        self.model = self.load_model()
        
    def load_model(self):
        
        if CtcConf.ascend_mode:
            from src.corrector.corrector import AclLiteModel
            model = AclLiteModel(CtcConf.csc_recall_model_fp_ascend_mode)
            logger.info('model loaded from: {}'.format(CtcConf.csc_recall_model_fp_ascend_mode))
        else:
            
            model = ModelingCTC.from_pretrained(self.in_model_dir)
            model.eval()
     
            if self.use_cuda and torch.cuda.is_available():
                if self.cuda_id is not None:
                    torch.cuda.set_device(self.cuda_id)
                model.cuda()

                if not self.onnx_mode:
                    model = model.half()  # 半精度
            logger.info('model loaded from: {}'.format(self.in_model_dir))

        return model
    
    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')
        
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        
        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id
    
    
    def id_list2ctag_list(self, id_list)->list:
        
        return [self._id2ctag[i] for i in id_list]
    
    
    @torch.no_grad()
    def predict(
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
        
        if CtcConf.ascend_mode:
            return self.predict_on_ascend_wrapper(texts, batch_size, return_topk=1)
        
        outputs = []
        if isinstance(texts, str):
            texts = [texts]

        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]
      
            # 将一些bert词典中不存在的 中文符号 转换为 英文符号
            # 加空格让bert对英文分字
            # batch_char_based_texts = [
            #     '始' + replace_punc_for_bert(text)
            #     for text in batch_texts
            # ]
            
            
            batch_char_based_texts = [
                replace_punc_for_bert(text)
                for text in batch_texts
            ]
            inputs = self.tokenizer(
                batch_char_based_texts, return_tensors='pt')

            # inputs['input_ids'][..., 1] = 1  # 把 始 换成 [unused1]
            if not CtcConf.ascend_mode:
                # 盒子模式使用onnx模型
                if self.use_cuda and torch.cuda.is_available():
                    for k,v in inputs.items():
                        inputs[k] = v.cuda()
                c_preds = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids'],
                )[0]
            else:
                c_preds = self.model.run(None, {
                    'input_ids': inputs['input_ids'].numpy(),
                    'attention_mask': inputs['attention_mask'].numpy(),
                    'token_type_ids': inputs['token_type_ids'].numpy(),
                })[0]
                preds = torch.from_numpy(preds)

            for idx, length in enumerate(inputs['length'].tolist()):
                true_idx = range(1, length - 1)
                # true_idx = range(0, length - 0)
                pred = c_preds[idx, true_idx, ...]
                pred_h = torch.softmax(pred, dim=-1)
                
                pred_h, pred = pred_h.topk(k=return_topk,
                                           dim=-1,
                                           largest=True,
                                           sorted=True)  # logit, idx
                pred_h = pred_h.tolist()
                
                pred = [self.id_list2ctag_list(x) for x in pred]
                pred = list(list(zip(*ele)) for ele in zip(pred, pred_h))
                origin_text = batch_texts[idx]
                # 还原空格

                [pred.insert(i, [(v, 1.0)]) for i, v in enumerate(
                    origin_text) if v in SPACE_SIGNS]
                
                # 把开头的占位符还原回来
                # origin_char_list = [''] + list(origin_text)
                origin_char_list = list(origin_text)
                outputs.append(list(zip(origin_char_list, pred)))
                
        return outputs
    
    
    
    def softmax(self, x, dim=None):
        x = x - x.max(axis=dim, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=dim, keepdims=True)
    
    def predict_on_ascend(
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
      
            # 将一些bert词典中不存在的 中文符号 转换为 英文符号
            # 加空格让bert对英文分字
            batch_char_based_texts = [
                '始' + replace_punc_for_bert(text)
                for text in batch_texts
            ]
            inputs = self.tokenizer(
                batch_char_based_texts, max_len=128,return_tensors='np')

            inputs['input_ids'][..., 1] = 1  # 把 始 换成 [unused1]
            X1, X2, X3 = inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]
            c_preds = self.model.execute([X1, X2, X3])[0]
            c_preds = torch.from_numpy(c_preds).float()
            for idx, length in enumerate(inputs['length'].tolist()):
                true_idx = range(1, length - 1)
                pred = c_preds[idx, true_idx, ...]
                pred_h = torch.softmax(pred, dim=-1)
                # "softmax_lastdim_kernel_impl" not implemented for 'Half'
                pred_i = torch.argmax(pred_h, axis=-1)
                pred_h = pred_h.tolist()
                
                pred = self.id_list2ctag_list(pred_i)
                pred = list([(tag, probs[idx])] for tag, idx, probs in zip(pred, pred_i, pred_h)) #暂时只取第一个
                origin_text = batch_texts[idx]
                # 还原空格

                [pred.insert(i, [(v, 1.0)]) for i, v in enumerate(
                    origin_text) if v in SPACE_SIGNS]
                
                # 把开头的占位符还原回来
                origin_char_list = [''] + list(origin_text)
                outputs.append(list(zip(origin_char_list, pred)))
                
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






if __name__ == '__main__':
    from configs.ctc_conf import CtcConf

    
    p = PredictorCsc(in_model_dir='model/miduCTC_v3.7.0_csc3model/ctc_csc', use_cuda=False)
    
    # '请问董秘公司海外电商渠道销售都有那些平台...'
    r = p.predict(['不能侵犯广大人喔群众的利益！'],  return_topk=3)
    print(r)
