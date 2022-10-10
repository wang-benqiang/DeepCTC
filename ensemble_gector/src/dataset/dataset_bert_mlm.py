import os
from sklearn.utils import shuffle

import torch
from logs import logger
from src.tokenizer.bert_tokenizer import CustomBertTokenizer
from torch.utils.data import Dataset
from utils.data_helper import replace_punc_for_bert, include_cn, inclue_punc
from utils.lmdb.db_manager import CtcDBManager
from utils.lmdb.yaoge_lmdb import TrainDataLmdb
import random
from difflib import SequenceMatcher
from utils.gector_preprocess_data import align_sequences
from transformers.data.data_collator import DataCollatorForWholeWordMask
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.models.bert.tokenization_bert import BertTokenizer


class DatasetMlmBert(Dataset):
    def __init__(self,
                 tokenizer: BertTokenizerFast,
                 max_seq_len,
                 max_dataset_len=-1,
                 db_dir='data/lm_db/db_test',
                 ctc_label_vocab_dir='src/vocab'
                 ):
        """

        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetMlmBert, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len
        self.tokenizer = tokenizer
        self.dataset_collate = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
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
    
    

    
    def convert_ids_to_ctags(self, ctag_id_list):
        return [self.tokenizer.convert_ids_to_tokens(i) if i!=self._loss_ignore_id else self._loss_ignore_id for i in ctag_id_list]
    

    
    @property
    def collate_fn(self):
        return self.dataset_collate.__call__
        
    
    def __getitem__(self, item):
        src_text, trg_text = self.db_manager.get_src_trg(item)
        
        while not include_cn(trg_text):
            # 如果数据中不包含文本.随机再抽一个数据
            src_text, trg_text = self.db_manager.get_src_trg(random.randint(0, self.data_len))

        inputs = self.parse_data(trg_text)

        return_dict = inputs

        return inputs

    def __len__(self):
        if self.max_dataset_len != -1 and self.max_dataset_len < self.data_len:
            return self.max_dataset_len
        return self.data_len

    def parse_data(self, text):

        
        # 随机在句子末尾增加或删除标点，增加鲁棒性
        punc_action = random.choice([True, False])

        if punc_action and len(text) > 3:
            if inclue_punc(text[-1]):
                text = text[:-1]
            else:
                # 随机选一个标点加到末尾
                random_punc = random.choice(self.punc_list_in_end)
                text = text+random_punc
        
        
        text = replace_punc_for_bert(
            text)[:self.max_seq_len - 2]
        # 对英文字级别打token，防止单词之类的#号bug

        inputs = self.tokenizer(text, padding='max_length', max_length=self.max_seq_len, truncation=True)
        # for k, v in inputs.items():
        #     inputs[k] = v[0]
      
    
        return inputs
    


if __name__ == '__main__':
    tokenizer_path = 'pretrained_model/chinese-macbert-large'

    # tokenizer = CustomBertTokenizer.from_pretrained(tokenizer_path)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    d = DatasetMlmBert(
        tokenizer=tokenizer,
        max_seq_len=128,
        max_dataset_len=1000,
        db_dir='data/lmdb/track2_cged_train',
    )
    # s = '望贵方快速回复'
    # # t = '望贵方尽快答复'
    # s = '众所都智，重视绿色食品越来越强调。'
    # t = '众所周知，现在越来越重视绿色食品。'

    # r = d.parse_data_by_gector(s, t)
    # r = d.gen_src_trg_iteration(s, t)
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=128, shuffle=True, collate_fn=d.collate_fn)
    for i in dataset:
        print(i)

