#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
from configs.ctc_conf import CtcConf
import torch
from src import logger
from src.deeplearning.modeling.modeling_lm import ModelLmCscBert
from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
from src.utils.data_helper import SPACE_SIGNS, include_cn, replace_punc_for_bert
from sklearn.metrics import classification_report


"csc task: detect sentence, token"
class PredictorCscLmBert:
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
        self._id2dtag, self._dtag2id = self.load_label_dict(
            ctc_label_vocab_dir)
        self.tokenizer = CustomBertTokenizer.from_pretrained(in_model_dir)
        self.use_cuda = use_cuda
        self.onnx_mode = onnx_mode
        self.batch_size = batch_size
        self.cuda_id = cuda_id
        self.model = self.load_model()
        
    def load_model(self):
        
        if CtcConf.ascend_mode:
            from src.corrector.corrector import AclLiteModel
            model = AclLiteModel(CtcConf.csc_recall_model_fp_ascend_mode)
            logger.info('model loaded from: {}'.format(CtcConf.csc_recall_model_fp_ascend_mode))
        else:
            
            model = ModelLmCscBert.from_pretrained(self.in_model_dir)
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

        
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

        return id2dtag, d_tag2id
    
    
    def id_list2dtag_list(self, id_list)->list:
        
        return [self._id2dtag[i] for i in id_list]
    
    
    @torch.no_grad()
    def predict(
        self,
        texts,
        return_topk=1,
        level ='all',
    ):
        """ seq2label

        Args:
            texts (list):
            ignore_idx_list: [[12], [13,14]]
        Returns:
            List[tuple]: [('你','$KEEP'), ('好', '$DELETE')],
        """
        
        if CtcConf.ascend_mode:
            return self.predict_on_ascend_wrapper(texts, self.batch_size, return_topk=1)
        
        token_outputs = []
        sent_outputs = []
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

            if not CtcConf.ascend_mode:
                
                if self.use_cuda and torch.cuda.is_available():
                    for k,v in inputs.items():
                        inputs[k] = v.cuda()
                sentence_h, token_h, total_loss = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                )
            else:
                sentence_h, token_h, total_loss = self.model.run(None, {
                    'input_ids': inputs['input_ids'].numpy(),
                    'attention_mask': inputs['attention_mask'].numpy(),
                })[0]
                preds = torch.from_numpy(preds)
            
            
            # detect sent level
            sent_h = torch.softmax(sentence_h, dim=-1)

            sent_h, sent_pred_idx = sent_h.topk(k=return_topk,
                                        dim=-1,
                                        largest=True,
                                        sorted=True)  # logit, idx
            sent_h = sent_h.tolist()
            
            sent_tags = [self.id_list2dtag_list(x) for x in sent_pred_idx]
                
            sent_outputs.extend(list(zip(batch_texts, sent_tags, sent_h)))
            for idx, length in enumerate(inputs['length'].tolist()):
                
           
                #  detect token level
                true_idx = range(1, length - 1)
                
                pred = token_h[idx, true_idx, ...]
            

                pred_h = torch.softmax(pred, dim=-1)
                
                pred_h, pred = pred_h.topk(k=return_topk,
                                           dim=-1,
                                           largest=True,
                                           sorted=True)  # logit, idx
                pred_prob_list = pred_h.tolist()
                
                pred_char_list = [self.id_list2dtag_list(x) for x in pred]
                
                origin_text = batch_texts[idx]
                # 还原空格

                [[pred_char_list.insert(i, [v]), pred_prob_list.insert(i, 1.0)] for i, v in enumerate(
                    origin_text) if v in SPACE_SIGNS]
                
                
                # origin_char_list = [''] + list(origin_text)
                origin_char_list = list(origin_text)
                token_outputs.append(list(zip(origin_char_list, pred_char_list, pred_prob_list)))
                
        return sent_outputs,token_outputs
    
    
    
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
            X1, X2 = inputs["input_ids"], inputs["attention_mask"]
            c_preds = self.model.execute([X1, X2])[0]
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




    
    def evaluate_f1(self, src_texts, trg_texts):
        
        sent_level_labels = [self._dtag2id['$ERROR'] if s!=t else self._dtag2id['$RIGHT']  for s,t in zip(src_texts, trg_texts) ]
        token_level_labels = [self._dtag2id['$ERROR'] if s_char!=t_char else self._dtag2id['$RIGHT']   for s,t in zip(src_texts, trg_texts) for s_char, t_char in zip(s,t)]
        
        sent_outputs, token_outputs = self.predict(src_texts, return_topk=1) 

        pred_sent_labels = [self._dtag2id[t[0]] for s, t, p in sent_outputs]
        pred_token_labels = [ self._dtag2id[t[0]] if include_cn(s) else self._dtag2id['$RIGHT']   for sent_token in token_outputs for s, t, p in sent_token]
        
        
        print('sent level report:')
        sent_report = classification_report(sent_level_labels, pred_sent_labels)
        print(sent_report)
        print('token level report:')
        token_report = classification_report(token_level_labels, pred_token_labels)
        print(token_report)
        
        
        print('end')
        
        
        
if __name__ == '__main__':
    from configs.ctc_conf import CtcConf

    
    p = PredictorCscLmBert(in_model_dir='model/csc_lm_bert_pretrain_2022Y06M26D15H/epoch1,ith_db1,step26369,test_sent_f1_97_04%,dev_sent_f1_98_89%', use_cuda=True)
    
    # '请问董秘公司海外电商渠道销售都有那些平台...'
    sent_outputs, token_outputs = p.predict(['我很不理姐这件事情。', '不能侵犯广大人是群众的利益！'],  return_topk=2)
    p.evaluate_f1(['我很不理姐这件事情。'], ['我很不理解这件事情。'])
    print(sent_outputs)
