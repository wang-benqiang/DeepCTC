import os
from sklearn.utils import shuffle

import torch
from logs import logger
from transformers.models.t5 import T5TokenizerFast
from torch.utils.data import Dataset
from utils.data_helper import replace_punc_for_bert, include_cn, inclue_punc
from utils.lmdb.db_manager import CtcDBManager
from utils.lmdb.yaoge_lmdb import TrainDataLmdb
import random


class DatasetCscT5(Dataset):
    def __init__(self,
                 tokenizer: T5TokenizerFast,
                 max_seq_len,
                 max_dataset_len=128,
                 db_dir='data/lm_db/db_test',
                 ):
        """

        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetCscT5, self).__init__()
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

        self.punc_list_in_end = ['.', '。', '。', '！', '!']
        self.punc_list_in_end = self.punc_list_in_end + \
            [i*2 for i in self.punc_list_in_end]  # 标点数量乘2

        # vocab id
        self._loss_ignore_id = -100

    def collate_fn(self, batch):
        # preprocess [(src, trg), (src, trg)]
        

        src_texts, trg_texts = [], []
        [[src_texts.append(src_text), trg_texts.append(trg_text)] for (src_text, trg_text) in batch]
        
        inputs = self.tokenizer(src_texts,
                                truncation=True,
                                padding=True,
                                max_length=self.max_seq_len,
                                return_tensors="pt",)
        labels = self.tokenizer(trg_texts,
                                truncation=True,
                                padding=True,
                                max_length=self.max_seq_len,
                                return_tensors="pt").input_ids
        labels = torch.where(labels==0, self._loss_ignore_id, labels)
        inputs['labels'] = labels
        return inputs
    def __getitem__(self, item):
        src_text, trg_text = self.db_manager.get_src_trg(item)

        while not include_cn(src_text):
            # 如果数据中不包含文本.随机再抽一个数据
            src_text, trg_text = self.db_manager.get_src_trg(
                random.randint(0, self.data_len))
        if len(src_text) != len(trg_text):
            src_text = trg_text

        

        src_text, trg_text = self.preprocess_text_pair(src_text, trg_text)


        return [src_text, trg_text]

    def __len__(self):
        if self.max_dataset_len != -1 and self.max_dataset_len < self.data_len:
            return self.max_dataset_len
        return self.data_len

    def preprocess_text_pair(self, src_text, trg_text):

        # 随机在句子末尾增加或删除标点，增加鲁棒性
        punc_action = random.choice([True, False])

        if punc_action and len(src_text) > 3:
            if inclue_punc(src_text[-1]):
                src_text, trg_text = src_text[:-1], trg_text[:-1]
            else:
                # 随机选一个标点加到末尾
                random_punc = random.choice(self.punc_list_in_end)
                src_text, trg_text = src_text+random_punc, trg_text+random_punc

        src_text, trg_text = replace_punc_for_bert(
            src_text)[:self.max_seq_len - 2], replace_punc_for_bert(trg_text)[:self.max_seq_len - 2]
      
        return src_text, trg_text


if __name__ == '__main__':
    tokenizer_path = 'pretrained_model/mengzi-t5-base '

    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)
    # tokenizer
    d = DatasetCscT5(
        tokenizer=tokenizer,
        max_seq_len=128,
        max_dataset_len=100000,
        db_dir='data/lmdb/merge_wiki_political_chengyu_6000w',
    )
    dataset = torch.utils.data.dataloader.DataLoader(
        d, batch_size=128, shuffle=True, collate_fn=d.collate_fn)
    for i in dataset:
        print(i)
