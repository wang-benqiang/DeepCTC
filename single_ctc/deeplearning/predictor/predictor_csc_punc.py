#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from src import logger
from src.deeplearning.modeling.modeling_csc_quant_punc import BertForCscQuantPunc
from src.utils.data_helper import SPACE_SIGNS, replace_punc_for_bert
import numpy as np
from configs.ctc_conf import CtcConf
from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer


class PredictorCscPunc:
    def __init__(
        self,
        pretrained_model_dir,
        use_cuda=True,
        cuda_id=None,
        onnx_mode=False
    ):

        self.pretrained_model_dir = pretrained_model_dir

        logger.info('model loaded from dir {}'.format(
            self.pretrained_model_dir))
        self.use_cuda = use_cuda
        self.cuda_id = cuda_id
        self.onnx_mode = onnx_mode
        self.tokenizer = CustomBertTokenizer.from_pretrained(pretrained_model_dir)
        self.model = self.load_model()
    
    def load_model(self):
        
        if CtcConf.ascend_mode:
            from src.corrector.corrector import AclLiteModel
            model = AclLiteModel(CtcConf.csc_punc_recall_model_fp_ascend_mode)
            logger.info('model loaded from: {}'.format(CtcConf.csc_punc_recall_model_fp_ascend_mode))

        else:
            
            model = BertForCscQuantPunc.from_pretrained(self.pretrained_model_dir)
            model.eval()
     
            if self.use_cuda and torch.cuda.is_available():
                if self.cuda_id is not None:
                    torch.cuda.set_device(self.cuda_id)
                model.cuda()

               
                model = model.half()  # 半精度
            logger.info('model loaded from: {}'.format(self.pretrained_model_dir))

        return model
    

    @torch.no_grad()
    def predict(self, texts, return_topk=1, batch_size=32):
        """ seq2label
        Args:
            texts (list): 
            ignore_eng: 是否忽略英文检错
        Returns:
            List[tuple]: [ ('中', [ ('中', 0.9977), ('众', 0.0023)] ) ],
        """
        if CtcConf.ascend_mode:
            return self.predict_on_ascend_wrapper(texts, batch_size, return_topk=1)
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
                replace_punc_for_bert(text)+'尾'
                for text in batch_texts
            ]
            inputs = self.tokenizer(batch_char_based_texts,
                                          return_tensors='pt')
            inputs['input_ids'][..., -2] = self.tokenizer.vocab['[unused60]']  # sep前面加了一个站位符,进一步深化生产经营效益.[unused60]
            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                inputs['token_type_ids'] = inputs['token_type_ids'].cuda()
             

            c_preds= self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
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
                pred = list(list(zip(*ele)) for ele in zip(pred, pred_h))
                origin_text = batch_texts[idx]+'尾'
           
                [pred.insert(i, [(v, 1.0)]) for i, v in enumerate(
                    origin_text) if v in SPACE_SIGNS] # 空字符,pred插入1概率的预测结果
                outputs.append(list(zip(origin_text, pred)))
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
                replace_punc_for_bert(text)+'尾'
                for text in batch_texts
            ]
            inputs = self.tokenizer(
                batch_char_based_texts, max_len=128,return_tensors='np')
            inputs['input_ids'][..., -2] = self.tokenizer.vocab['[unused60]']  # sep前面加了一个站位符,进一步深化生产经营效益.[unused60]

            inputs['input_ids'][..., 1] = 1  # 把 始 换成 [unused1]
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
                
                pred = [self.tokenizer.convert_ids_to_tokens(
                    x) for x in pred_i]
                pred = list([(tag, probs[idx])] for tag, idx, probs in zip(pred, pred_i, pred_h)) #暂时只取第一个
                origin_text = batch_texts[idx]
                # 还原空格

                [pred.insert(i, [(v, 1.0)]) for i, v in enumerate(
                    origin_text) if v in SPACE_SIGNS]
                # 把开头的占位符还原回来
                origin_char = origin_text+'尾'
                outputs.append(list(zip(origin_char, pred)))
                
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
    model_dir = 'model/punc_model'
    p = PredictorCscPunc(pretrained_model_dir=model_dir)
    r = p.predict(['他说，中石化（香港）海南石油公司是洋浦的重点企业〕希望下一步中石化（香港海南石油公司继续谋求更大发展,为海南自贸港建设作出更大贡献。工委管委会也将一如既往给予全力支持，研究解决企业发展难题,助力企业发展壮大｝周军平指出，今年是十四五的开局之年，洋浦将朝着更高的发展目标迈进，希望中石化（香港）海南石油公司能够与工委管委会共同发力，开足马力～继续做大进出口贸易和离岸贸易。同时充分结合自贸港政策的实施,谋划好二期项目建设？相关部门要予以配合支持。要持续开展制度集成创新，进一步深化生产经营效益.'])
    r = p.predict(['“你一定要保重身体,小明淡淡地说道.'])
    print(r)