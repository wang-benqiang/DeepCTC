import gc
import os

import torch
from logs import logger
from src.tokenizer.bert_tokenizer import CustomBertTokenizer
from torch.utils.data import Dataset
from utils.data_helper import replace_punc_for_bert
from utils.lmdb.db_manager import CtcDBManager
from utils.lmdb.yaoge_lmdb import TrainDataLmdb
import random


class DatasetGpt(Dataset):
    def __init__(self,
                 tokenizer: CustomBertTokenizer,
                 max_seq_len,
                 max_dataset_len=-1,
                 db_dir='data/lm_db/db_test',
                 db_gb_size=20,
                 ):
        """

        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetGpt, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len
        self.tokenizer = tokenizer

        self.db_dir = db_dir
        if self.db_dir is not None and os.path.exists(self.db_dir):
            
            
            if 'yaoge' in self.db_dir:
                self.db_manager = TrainDataLmdb(lmdb_dir=self.db_dir)
            else:
                
                self.db_manager = CtcDBManager(lmdb_dir=self.db_dir)
        

        self.data_len = len(self.db_manager)
        logger.info('all samples loaded, num: {}'.format(
            self.data_len))

        gc.collect()

    def __getitem__(self, item):
        src, trg = self.db_manager.get_src_trg(item)
        if len(src) != len(trg):
            src = trg
        
        # random select between pos text pair and neg text pair 
        # neg_sample_prob = random.random()
        # src = trg if neg_sample_prob >0.5 else src
        
        inputs = self.parse_line_data_and_label(trg, trg)

        return_dict = {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
            'labels': torch.LongTensor(inputs['labels']),

        }

        return return_dict

    def __len__(self):
        if self.max_dataset_len != -1 and self.max_dataset_len < self.data_len:
            return self.max_dataset_len
        return self.data_len

    def parse_line_data_and_label(self, src_text, trg_text):

        src, trg = replace_punc_for_bert(
            src_text)[:self.max_seq_len - 2], replace_punc_for_bert(trg_text)[:self.max_seq_len - 2]
        # 对英文字级别打token，防止单词之类的#号bug

        inputs = self.tokenizer(src, max_len=self.max_seq_len)

        if src_text != trg_text:
            inputs_trg = self.tokenizer(trg, max_len=self.max_seq_len)
            inputs['labels'] = inputs_trg['input_ids']
        else:
            inputs['labels'] = inputs['input_ids']
            
        # 取单个value
        for k, v in inputs.items():
            inputs[k] = v[0]
            
        return inputs


if __name__ == '__main__':
    tokenizer_path = 'pretrained_model/gpt2_cn_cluesmall'

    tokenizer = CustomBertTokenizer.from_pretrained(tokenizer_path)
    d = DatasetGpt(
        tokenizer=tokenizer,
        max_seq_len=128,
        max_dataset_len=512,
        db_dir='data/lmdb/merge_wiki_political_chengyu_6000w',
    )
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=10)
    for i in dataset:
        print(i)

