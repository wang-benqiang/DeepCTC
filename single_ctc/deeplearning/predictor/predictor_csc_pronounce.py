#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
from configs.ctc_conf import CtcConf
import torch
from src.deeplearning.modeling.modeling_csc_pronounce import ModelingCscPronounce
from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
from src.utils.data_helper import SPACE_SIGNS, replace_punc_for_bert
from src.utils.data_helper import (include_cn, inclue_punc,
                                   replace_punc_for_bert)
from src import logger
import json

class PredictorCscPronounce:
    def __init__(
        self,
        in_model_dir,
        ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
        use_cuda=True,
        cuda_id=None,
        onnx_mode=False
    ):

        self.in_model_dir = in_model_dir
        self.id2dtag, self.dtag2id, self.id2ctag, self.ctag2id, self.char2shengmu_dict, self.char2yunmu_dict, self.char_pronounce_confusion = self.load_label_dict(
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
            
            model = ModelingCscPronounce.from_pretrained(self.in_model_dir)
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
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_2tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')
        
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        
        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        
        
        # char to pinyin code
        char2shengmu_dict = json.load(open('src/data/pinyin_data/char_shengmu.json', 'r', encoding='utf8'))
        char2yunmu_dict = json.load(open('src/data/pinyin_data/char_yunmu.json', 'r', encoding='utf8'))
        char_pronounce_confusion = json.load(open('src/data/pinyin_data/char_pronounce_confusion.json', 'r', encoding='utf8'))
        
        
        char2shengmu_dict['[UNK]']
        # pinyin code to id 
        pycode2id = {
            str(v):idx for idx,v in enumerate(list(range(0, 10)) + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        }
        length = len(pycode2id)
        pycode2id['[PAD]'] = length
        pycode2id['[UNK]'] = length+1
        
        char2shengmu_dict = {char:pycode2id.get(code, pycode2id['[UNK]']) for char, code in  char2shengmu_dict.items()}
        char2yunmu_dict = {char:pycode2id.get(code, pycode2id['[UNK]']) for char, code in  char2yunmu_dict.items()}
      
        logger.info('d_tag num: {}, detect_labels:{}'.format(len(id2dtag), d_tag2id))
        
        return id2dtag, d_tag2id, id2ctag, c_tag2id, char2shengmu_dict, char2yunmu_dict, char_pronounce_confusion
    
    def id_list2ctag_list(self, id_list)->list:
        
        return [self.id2ctag[i] for i in id_list]
    
    def id_list2dtag_list(self, id_list)->list:
        
        return [self.id2dtag[i] for i in id_list]
    
    def get_pinyin_info(self, texts, max_seq_len, return_tensors='pt'):
        
        input_shengmu, input_yunmu = [], []
        
        if isinstance(texts, str):
            texts = texts
        
        for text in texts:
            ignore_loss_seq_len= max_seq_len-(len(text)+1)  # sep and pad
            
            text_shengmu = [self.char2shengmu_dict['[PAD]']] + [self.char2shengmu_dict.get(char, self.char2shengmu_dict['[UNK]']) if  include_cn(char) else self.char2shengmu_dict['[UNK]'] for char in text] + [self.char2shengmu_dict['[PAD]']] * ignore_loss_seq_len

            text_yunmu = [self.char2yunmu_dict['[PAD]']] + [self.char2yunmu_dict.get(char, self.char2yunmu_dict['[UNK]']) if  include_cn(char) else self.char2yunmu_dict['[UNK]'] for char in text] + [self.char2yunmu_dict['[PAD]']] * ignore_loss_seq_len

            input_shengmu.append(text_shengmu)
            input_yunmu.append(text_yunmu)
            
        if return_tensors =='pt':
            input_shengmu = torch.LongTensor(input_shengmu)
            input_yunmu = torch.LongTensor(input_yunmu)
        return input_shengmu, input_yunmu
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
            
            
            batch_char_based_texts = [
                replace_punc_for_bert(text)
                for text in batch_texts
            ]
            inputs = self.tokenizer(
                batch_char_based_texts, return_tensors='pt')
            max_seq_len = inputs['input_ids'].shape[-1]
            inputs['input_shengmu'], inputs['input_yunmu'] = self.get_pinyin_info(batch_char_based_texts, max_seq_len, return_tensors='pt')
            
            if self.use_cuda and torch.cuda.is_available():
                for k,v in inputs.items():
                    inputs[k] = v.cuda()
            model_preds = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
                input_shengmu=inputs['input_shengmu'],
                input_yunmu=inputs['input_yunmu'],
            )

            batch_c_preds, batch_d_preds = model_preds[0], model_preds[1]

            for idx, length in enumerate(inputs['length'].tolist()):
                true_idx = range(1, length - 1)
                # true_idx = range(0, length - 0)
                c_preds = batch_c_preds[idx, true_idx, ...]
                d_preds = batch_d_preds[idx, true_idx, ...]
                
                c_pred_prob = torch.softmax(c_preds, dim=-1)
                d_pred_prob = torch.softmax(d_preds, dim=-1)
                c_pred_prob[..., 1] -= d_pred_prob[..., 1] # care detect output
                c_pred_prob, c_pred_idx = c_pred_prob.topk(k=return_topk,
                                           dim=-1,
                                           largest=True,
                                           sorted=True)  # logit, idx
                
                # d_pred_prob, d_pred_idx = d_pred_prob.topk(k=2,
                #                            dim=-1,
                #                            largest=True,
                #                            sorted=True)  # logit, idx
                
                c_pred_prob = c_pred_prob.tolist()
                # d_pred_prob = d_pred_prob.tolist()
                
                c_pred_tag = [self.id_list2ctag_list(x) for x in c_pred_idx]
                # d_pred_tag = [self.id_list2dtag_list(x) for x in d_pred_idx]
                
                c_pred_tag_prob = list(list(zip(*ele)) for ele in zip(c_pred_tag, c_pred_prob))
                # d_pred_tag_prob = list(list(zip(*ele)) for ele in zip(d_pred_tag, d_pred_prob))
                origin_text = batch_texts[idx]
                # 还原空格

                [c_pred_tag_prob.insert(i, [(v, 1.0)]) for i, v in enumerate(
                    origin_text) if v in SPACE_SIGNS]
                
                # [d_pred_tag_prob.insert(i, [(v, 1.0)]) for i, v in enumerate(
                #     origin_text) if v in SPACE_SIGNS]
                
                # 把开头的占位符还原回来
                # origin_char_list = [''] + list(origin_text)
                origin_char_list = list(origin_text)
                outputs.append(list(zip(origin_char_list, c_pred_tag_prob)))
                # outputs.append(list(zip(origin_char_list, c_pred_tag_prob, d_pred_tag_prob)))
                
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
    in_model_dir = CtcConf.csc_recall_model_dir3
    in_model_dir = 'model/csc_pronounce_pretrain_27e_2022Y07M04D18H/epoch2,ith_db0,step1,testf1_87_14%,devf1_98_29%'
    p = PredictorCscPronounce(in_model_dir, use_cuda=True)
    
    # '请问董秘公司海外电商渠道销售都有那些平台...'
    r = p.predict(['这一场欢天喜地的乡村舞开始了，这是世界上所有跳舞你边最好的一种跳舞。'],  return_topk=3)
    print(r)
