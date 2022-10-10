#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import re
from functools import reduce
import onnxruntime as ort
import torch
from src import logger
from src.deeplearning.modeling.modeling_seq2label import Sequence2Label
from src.utils.data_helper import (SPACE_SIGNS, include_cn,
                                   replace_punc_for_bert)
from transformers.models.bert import BertTokenizer
from transformers.models.electra import ElectraTokenizer
from tqdm import tqdm
"暂时用于多字少字检测"


class PredictorSeq2label:
    def __init__(
        self,
        pretrained_model_dir,
        ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
        pretrained_model_type='electra',
        d_tag_type='all',
        use_cuda=True,
        cuda_id=None,
        onnx_mode=False
    ):

        self.d_tag_type = d_tag_type
        self.pretrained_model_dir = pretrained_model_dir
        self.use_cuda = use_cuda

        self.pretrained_model_type = pretrained_model_type
        self.eng_digit_re = re.compile(r'[a-zA-Z]|[0-9]')
        self.onnx_mode = onnx_mode
        self.cuda_id = cuda_id
        d_tag_types = ('all', 'redundant', 'miss')
        assert self.d_tag_type in d_tag_types, 'd_tag_type not in {}'.format(
            d_tag_types)

        if self.pretrained_model_type == 'electra':
            self.tokenizer = ElectraTokenizer.from_pretrained(
                pretrained_model_dir)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                pretrained_model_dir)

        self.id2dtag, self.d_tag2id = self.load_label_dict(
            ctc_label_vocab_dir)
        self.model = self.load_model()
        if self.onnx_mode:
            # try:
            self.model_to_onnx()
        # except Exception as e:
            # logger.exception(e)
            # logger.error('onnx failed! ')
            # self.onnx_mode = False
            # self.model = self.load_model()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # clear cuda cache

    def load_model(self):
        
        model = Sequence2Label.from_pretrained(self.pretrained_model_dir)
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

        if self.d_tag_type == 'all':
            tag_file_name = 'd_3tags.txt'
        elif self.d_tag_type == 'redundant':
            tag_file_name = 'redundant_tags.txt'
        elif self.d_tag_type == 'miss':
            tag_file_name = 'missing_tags.txt'
        dtag_fp = '{}/{}'.format(ctc_label_vocab_dir, tag_file_name)
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        print('d_tag num: {}, d_tag:{}'.format(len(id2dtag), id2dtag))

        return id2dtag, d_tag2id

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
        logit_threshold=0.5,
        only_care_cn=True,
        texts_ignore_idx_list=None,
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
        outputs = []
        if isinstance(texts, str):
            texts = [texts]

        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]
            if texts_ignore_idx_list is not None:
                batch_texts_ignore_idx_list = texts_ignore_idx_list[
                    start_idx:start_idx + batch_size]
            # 将一些bert词典中不存在的 中文符号 转换为 英文符号
            # 加空格让bert对英文分字
            batch_char_based_texts = [
                '始' + replace_punc_for_bert(text)
                for text in batch_texts
            ]

            inputs = self.tokenize_inputs(
                batch_char_based_texts, return_tensors='pt')

            inputs['input_ids'][..., 1] = 2  # 这里将 始 替换为bert词典的[unused2]
            if not self.onnx_mode:
                if self.use_cuda and torch.cuda.is_available():
                    inputs['input_ids'] = inputs['input_ids'].cuda()
                    inputs['attention_mask'] = inputs['attention_mask'].cuda()
                    inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

                preds = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids'],
                )
            else:
                preds = self.model.run(None, {
                    'input_ids': inputs['input_ids'].numpy(),
                    'attention_mask': inputs['attention_mask'].numpy(),
                    'token_type_ids': inputs['token_type_ids'].numpy(),
                })[0]
                preds = torch.from_numpy(preds)

            for idx, length in enumerate(inputs['length'].tolist()):
                true_idx = range(1, length - 1)
                pred = preds[idx, true_idx, ...]
                pred = torch.softmax(pred, dim=-1).tolist()
                if self.d_tag_type in ('miss', 'redundant'):
                    # 设置阈值
                    pred = list(
                        map(lambda i: (self.id2dtag[1], i[1]) if i[1] >= logit_threshold else (self.id2dtag[0], i[0]), pred))
                origin_text_list = ['[unused2]'] + list(batch_texts[idx])
                # 把原来文本中的空格还原到pred中，

                [
                    pred.insert(i, '$KEEP')
                    for i, v in enumerate(origin_text_list) if v in SPACE_SIGNS
                ]
                if texts_ignore_idx_list is not None:
                    for ignore_idx in batch_texts_ignore_idx_list[idx]:
                        # 别字模型修改过的索引，多字少字不再对其操作
                        # 因为开头有unused所有+1
                        if pred[ignore_idx + 1] != '$KEEP':
                            print(
                                'igonre: idx:{}  char: {}, label restore from {} to $KEEP'
                                .format(ignore_idx,
                                        origin_text_list[ignore_idx+1],
                                        pred[ignore_idx + 1]))
                            pred[ignore_idx + 1] = ('$KEEP', 'igonre')
                output = list(zip(origin_text_list, pred))
                if only_care_cn:
                    if self.d_tag_type in ('miss'):
                        output = list(
                            map(
                                lambda char_dtag: (
                                    char_dtag[0], ('$KEEP', 'no_cn'))
                                if not include_cn(char_dtag[0]) and char_dtag[0] != '[unused2]' else char_dtag,
                                output))
                    else:
                        output = list(
                            map(
                                lambda char_dtag: (
                                    char_dtag[0], ('$KEEP', 'no_cn'))
                                if not include_cn(char_dtag[0]) else char_dtag,
                                output))
                outputs.append(output)
        return outputs

    @staticmethod
    def mask_str_to_list(text):
        """[summary]

        Args:
            text ([type]): '这 是一个[MASK]啊 ,你的[MASK]呢'

        Returns:
            [type]: ['这', ' ', '是', '一', '个', '[MASK]', '啊', ' ', ',', '你', '的', '[MASK]', '呢']
        """
        texts = text.split('[MASK]')
        if len(texts) < 2:
            return list(text)
        return list(reduce(lambda a, b: list(a) + ['[MASK]'] + list(b), texts))

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

    p = PredictorSeq2label(
        pretrained_model_dir='model/miduCTC_v2.3_0113/rdt',
        ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
        d_tag_type='redundant',
        onnx_mode=False,
    )
    import time
    p.convert_to_jit()
    t1 = time.time()
    for i in tqdm(range(300)):

        r = p.predict(['这里多了一个一个字这里多了一个一个字这里多了一个一个字这里多了一个一个字这里多了一个一个字']*16)
        
    t2 = time.time()
    print('cost:', t2-t1)
    print('cost one:', (t2-t1) / 1000)
    # print(r)
#     cost: 3.016176700592041
# cost one: 0.03016176700592041
# cost: 1.2423839569091797
# cost one: 0.012423839569091797

# cost: 8.645023584365845
# cost one: 0.008645023584365846

# cost: 28.88184881210327
# cost one: 0.02888184881210327
