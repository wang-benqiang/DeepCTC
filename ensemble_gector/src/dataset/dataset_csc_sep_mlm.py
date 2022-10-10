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


class DatasetCscSeq2Edit(Dataset):
    def __init__(self,
                 tokenizer: CustomBertTokenizer,
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
        super(DatasetCscSeq2Edit, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len
        self.tokenizer = tokenizer
        self.id2dtag, self.dtag2id, self.id2ctag, self.ctag2id = self.load_label_dict(
            ctc_label_vocab_dir)
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
        self._keep_d_tag_id, self._error_d_tag_id = self.dtag2id['$RIGHT'], self.dtag2id['$ERROR']
        self._keep_c_tag_id = self.ctag2id['$KEEP']
        self._delete_c_tag_id = self.ctag2id['$DELETE']
        self.replace_unk_c_tag_id = self.ctag2id['[REPLACE_UNK]']
        self._loss_ignore_id = -100
    
    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')
        
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        
        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id
    
    @staticmethod
    def match_ctc_idx(src_text, trg_text):
        keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list = [], [], [], []
        
        r = SequenceMatcher(None, src_text, trg_text)
        diffs = r.get_opcodes()

        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            
      
            if tag == 'replace' and i2-i1 == j2-j1:
                # 如果文本中出现连续不同的错误label，会被diff库直接处理成replace操作
                # 所以暂时确保只有是错字类型再输出label
                replace_idx_list += [ (i, '$REPLACE_'+trg_text[j]) for i, j in zip(range(i1, i2), range(j1, j2))]

            

        return keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list
    
    
    def __getitem__(self, item):
        src_text, trg_text = self.db_manager.get_src_trg(item)
        
        while not include_cn(src_text):
            # 如果数据中不包含文本.随机再抽一个数据
            src_text, trg_text = self.db_manager.get_src_trg(random.randint(0, self.data_len))
        if len(src_text) != len(trg_text):
            src_text = trg_text
        
        # random select between pos text pair and neg text pair 
       
        
        inputs = self.parse_line_data_and_label(src_text, trg_text)

        return_dict = {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
            'detect_labels': torch.LongTensor(inputs['d_tags']),
            'correct_labels': torch.LongTensor(inputs['c_tags'])
        }

        return return_dict

    def __len__(self):
        if self.max_dataset_len != -1 and self.max_dataset_len < self.data_len:
            return self.max_dataset_len
        return self.data_len

    def parse_line_data_and_label(self, src_text, trg_text):

        
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
        # 对英文字级别打token，防止单词之类的#号bug

        inputs = self.tokenizer(src_text, max_len=self.max_seq_len)
        for k, v in inputs.items():
            inputs[k] = v[0]
        _, replace_idx_list, _, _ = self.match_ctc_idx(src_text, trg_text)
         # --- 对所有 token 计算loss ---
        src_len = len(src_text)
        ignore_loss_seq_len= self.max_seq_len-(src_len+1)  # sep and pad
        # 先默认给keep，会面对有错误标签的进行修改
        d_tags = [self._loss_ignore_id] + [self._keep_d_tag_id if not inclue_punc(c) else self._loss_ignore_id for c in src_text] + [self._loss_ignore_id] * ignore_loss_seq_len
        c_tags = [self._loss_ignore_id] + [self._keep_c_tag_id if not inclue_punc(c) else self._loss_ignore_id for c in src_text] + [self._loss_ignore_id] * ignore_loss_seq_len
        
        
        for (replace_idx, replace_char) in replace_idx_list:
            # +1 是因为input id的第一个字是cls
            d_tags[replace_idx+1] = self._error_d_tag_id
            c_tags[replace_idx+1] = self.ctag2id.get(replace_char, self.replace_unk_c_tag_id)
        
        
        inputs['d_tags'] = d_tags
        inputs['c_tags'] = c_tags
        return inputs


if __name__ == '__main__':
    tokenizer_path = 'pretrained_model/gpt2_cn_cluesmall'

    tokenizer = CustomBertTokenizer.from_pretrained(tokenizer_path)
    d = DatasetCscSeq2Edit(
        tokenizer=tokenizer,
        max_seq_len=128,
        max_dataset_len=100000,
        db_dir='data/lmdb/merge_wiki_political_chengyu_6000w',
    )
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=128, shuffle=True)
    for i in dataset:
        print(i)

