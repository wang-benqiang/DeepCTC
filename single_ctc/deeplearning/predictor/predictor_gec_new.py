#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from functools import reduce
import onnxruntime as ort
import torch
from src import logger
from src.deeplearning.modeling.modeling_gec_new import ModelingGEC
from src.utils.data_helper import (SPACE_SIGNS_2, replace_punc_for_bert_keep_space)
from transformers.models.electra import ElectraTokenizer
"暂时用于多字少字检测"


class PredictorGEC:
    def __init__(
        self,
        pretrained_model_dir,
        use_cuda=True,
        cuda_id=None,
        onnx_mode=False
    ):

        self.pretrained_model_dir = pretrained_model_dir
        self.use_cuda = use_cuda

        self.onnx_mode = onnx_mode
        self.cuda_id = cuda_id
     
        
        self.tokenizer = ElectraTokenizer.from_pretrained(
            pretrained_model_dir)

        self.model = self.load_model()
        if self.onnx_mode:
            self.model_to_onnx()
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # clear cuda cache

    def load_model(self):
        
        model = ModelingGEC.from_pretrained(self.pretrained_model_dir)
        model.eval()
        if self.use_cuda and torch.cuda.is_available():
            if self.cuda_id is not None:
                torch.cuda.set_device(self.cuda_id)
            model.cuda()

            if not self.onnx_mode:
                model = model.half()  # 半精度
        logger.info('model loaded from: {}'.format(self.pretrained_model_dir))

        return model



    def tokenize_inputs(self, texts, return_tensors=None):
        "预测tokenize, 按batch texts中最大的文本长度来pad, realise只需要input id, mask, length"

        cls_id, sep_id, pad_id, unk_id = self.tokenizer.vocab['[CLS]'], self.tokenizer.vocab[
            '[SEP]'], self.tokenizer.vocab['[PAD]'], self.tokenizer.vocab['[UNK]']
        input_ids, attention_mask, token_type_ids, length = [], [], [], []
        max_len = max([len(text) for text in texts]) + 2  # 注意+2

        for text in texts:
            true_input_id = [self.tokenizer.vocab.get(
                c, unk_id) for c in text][:max_len-2]
            pad_len = (max_len-len(true_input_id)-2)
            input_id = [cls_id] + true_input_id + [sep_id] + [pad_id] * pad_len

            a_mask = [1] * (len(true_input_id) + 2) + [0] * pad_len
            token_type_id = [0] * max_len
            input_ids.append(input_id)
            attention_mask.append(a_mask)
            token_type_ids.append(token_type_id)
            length.append(sum(a_mask))

        if return_tensors == 'pt':
            return {'input_ids': torch.LongTensor(input_ids),
                    'attention_mask': torch.LongTensor(attention_mask),
                    'token_type_ids': torch.LongTensor(token_type_ids),
                    'length': torch.LongTensor(length)}
        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'length': length}

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
        outputs = []
        if isinstance(texts, str):
            texts = [texts]

        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]
      
            # 将一些bert词典中不存在的 中文符号 转换为 英文符号
            # 加空格让bert对英文分字
            batch_char_based_texts = [
                '始' + replace_punc_for_bert_keep_space(text)
                for text in batch_texts
            ]
            inputs = self.tokenize_inputs(
                batch_char_based_texts, return_tensors='pt')

            inputs['input_ids'][..., 1] = 1  # 把 始 换成 [unused1]
            if not self.onnx_mode:
                if self.use_cuda and torch.cuda.is_available():
                    inputs['input_ids'] = inputs['input_ids'].cuda()
                    inputs['attention_mask'] = inputs['attention_mask'].cuda()
                    inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

                d_preds, c_preds, loss = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids'],
                )
            else:
                d_preds, c_preds, loss = self.model.run(None, {
                    'input_ids': inputs['input_ids'].numpy(),
                    'attention_mask': inputs['attention_mask'].numpy(),
                    'token_type_ids': inputs['token_type_ids'].numpy(),
                })[0]
                preds = torch.from_numpy(preds)

            for idx, length in enumerate(inputs['length'].tolist()):
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
                origin_text = batch_texts[idx]
                # 还原空格

                [pred.insert(i, [(v, 1.0)]) for i, v in enumerate(
                    origin_text) if v in SPACE_SIGNS_2]
                
                # 把开头的占位符还原回来
                origin_char_list = [''] + list(origin_text)
                outputs.append(list(zip(origin_char_list, pred)))
                
        return outputs



    def model_to_onnx(self):
        
        self._onnx_save_fp = os.path.join(
            self.pretrained_model_dir, 'model_fp16.onnx')
        dummpy_inputs = {
            'input_ids': torch.randint(1, 5, (1, 128)).cuda(),
            'attention_mask': torch.randint(0, 1, (1, 128)).cuda(),
            'token_type_ids': torch.randint(0, 1, (1, 128)).cuda(),
        }
        
        with torch.no_grad():
            symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}  # 指定可变维度
            torch.onnx.export(
                self.model,
                (
                    dummpy_inputs['input_ids'],
                    dummpy_inputs['attention_mask'],
                    dummpy_inputs['token_type_ids'],
                ),

                self._onnx_save_fp,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids',
                             'attention_mask',
                             'token_type_ids'],
                output_names=['sequence_output'],
                dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                              'attention_mask': symbolic_names,
                              'token_type_ids': symbolic_names,
                              'sequence_output': symbolic_names,
                              }

            )

        self.model = self.load_onnx_model()

    def convert_to_jit(self):

        self.model = torch.jit.trace(self.model, (torch.randint(0, 1, (3, 122)).cuda(),
                                                  torch.randint(0, 1, (3, 122)).cuda(),
                                                  torch.randint(0, 1, (3, 122)).cuda()), strict=False)
        # torch.jit.save(self.model, 'model/jit.pt')
        # self.model = torch.jit.load('model/jit.pt')
        logger.info('bert layer -> jit format')

    def load_onnx_model(self):
        
        # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.onnx_providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            # 'CPUExecutionProvider',
        ]
        # To enable model serialization and store the optimized graph to desired location.
        # sess_options.optimized_model_filepath = os.path.join(
        #     self._onnx_save_fp, "optimized_model_gpu.onnx")
        # sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
        # session = ort.InferenceSession(self._onnx_save_fp, sess_options, providers=providers)
        session = ort.InferenceSession(
            self._onnx_save_fp, providers=self.onnx_providers)
        
        dummpy_inputs = {
            'input_ids': torch.randint(1, 5, (2, 122)).numpy(),
            'attention_mask': torch.randint(0, 1, (2, 122)).numpy(),
            'token_type_ids': torch.randint(0, 1, (2, 122)).numpy(),
        }
        # 预热一次
        preds = session.run(None, {
            'input_ids': dummpy_inputs['input_ids'],
            'attention_mask': dummpy_inputs['attention_mask'],
            'token_type_ids': dummpy_inputs['token_type_ids'],
        })[0]

        return session


if __name__ == '__main__':
    predictor = PredictorGEC(pretrained_model_dir='model/gec_new_base_pretrain_2022Y03M29D14H/epoch1,ith_db1,step183,testf1_0%,devf1_0%',
                             use_cuda=True)
    texts = ['今天今天的天气真不呀,']
    
    r = predictor.predict(texts, return_topk=3)
    
    print(r)
    
    
    