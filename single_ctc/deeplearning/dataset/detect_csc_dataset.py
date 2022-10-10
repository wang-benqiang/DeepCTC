import gc
import os
import pickle
import random
from difflib import SequenceMatcher

import torch
from rich.progress import track
from src import logger
from src.deeplearning.dataset.lm_db.db_manager import CtcDBManager
from src.utils.data_helper import replace_punc_for_bert
from torch.utils.data import Dataset
from transformers import BertTokenizer
from src.deeplearning.dataset.lm_db.yaoge_lmdb import TrainDataLmdb



class DatasetDetectCsc(Dataset):
    def __init__(self,
                 data_dir_list,
                 tokenizer: BertTokenizer,
                 max_seq_len,
                 ctc_label_vocab_dir,
                 data_file_num_range_list=None,
                 max_dataset_len=None,
                 pos_sample_gen_ratio=0.5,
                 ds_pkl_fp=None,
                 db_dir='src/deeplearning/dataset/lm_db/db/detect_csc',
                 db_gb_size=20,
                 ):
        """
        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetDetectCsc, self).__init__()
        self.data_dir_list = data_dir_list
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len
        self.data_file_num_range_list = data_file_num_range_list
        self.ds_pkl_fp = ds_pkl_fp
        self.tokenizer = tokenizer
        self.pos_sample_gen_ratio = pos_sample_gen_ratio
        self.id2dtag, self.dtag2id = self.load_label_dict(
            ctc_label_vocab_dir)

        self.dtag_num = len(self.dtag2id)

        self.db_dir = db_dir
        if os.path.exists(self.db_dir):
            # self.db_manager = CtcDBManager(lmdb_dir=db_dir) 
            "yaoge"
            if 'yaoge' in db_dir:
                self.db_manager = TrainDataLmdb(lmdb_dir=db_dir)
            else:
                self.db_manager = CtcDBManager(lmdb_dir=db_dir)
        else:
            all_src, all_trg = self._read_rdt_data_from_file()
            logger.info('neg samples loaded, num: {}'.format(
                len(all_src)))
            pos_samples = self.gen_pos_data(all_trg)
            logger.info('pos samples loaded, num: {}'.format(
                len(pos_samples)))
            if len(pos_samples) > 0:
                all_src.extend(pos_samples), all_trg.extend(pos_samples)
            self.db_manager = CtcDBManager(lmdb_dir=db_dir)
            self.db_manager.create_db(all_src, all_trg,  db_gb_size=db_gb_size)

        # self.data_len = self.db_manager.get_data_len()
        self.data_len = len(self.db_manager)

        logger.info('all samples loaded, num: {}'.format(
            self.data_len))

        all_src, all_trg = [], []
        gc.collect()
        if self.ds_pkl_fp is not None:
            self.save_ds_pkl()
            logger.info('dataset pkl saved at {}'.format(self.ds_pkl_fp))

    def load_label_dict(self, ctc_label_vocab_dir):

        dtag_fp = '{}/{}'.format(ctc_label_vocab_dir, 'replace_tags.txt')
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id,

    def __getitem__(self, item):
        src, trg = self.db_manager.get_src_trg(item)
        inputs = self.parse_line_data_and_label(src, trg)
        return {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
            'd_tags': torch.LongTensor(inputs['d_tags'])
        }

    def __len__(self):
        if self.max_dataset_len is not None and self.max_dataset_len<self.data_len:
            return self.max_dataset_len
        return self.data_len

    def _read_rdt_data_from_file(self):
        """读取ctc数据

        Args:
            keep_one_append ([type]): 是否迭代生成多条数据
            chunk_num (int, optional): [description]. Defaults to 100000.

        Returns:
            list:[ [['$START', '$KEEP'], ['明', '$KEEP']], ...]
        """
        all_file_fp = []
        error_line_num = 0
        for data_dir in self.data_dir_list:
            current_dir_files = os.listdir(data_dir)
            current_dir_file_fp = [
                '{}/{}'.format(data_dir, f_name) for f_name in current_dir_files
            ]
            all_file_fp.extend(current_dir_file_fp)
        all_src, all_trg = [], []

        if self.data_file_num_range_list is not None:
            filter_all_file_fp = []
            fp_allowed_num_list = [i for i in range(
                self.data_file_num_range_list[0], self.data_file_num_range_list[1])]
            for i in all_file_fp:
                try:
                    file_num = int(i.split('.')[-2].split('_')[-1])
                    if file_num in fp_allowed_num_list:
                        filter_all_file_fp.append(i)
                except:
                    # 有些没有编号则直接选择
                    filter_all_file_fp.append(i)
            all_file_fp = filter_all_file_fp
        print('loading {} files:{}'.format(len(all_file_fp), str(all_file_fp)))
        for file_fp in track(all_file_fp,
                             description='Processing raw data...',
                             total=len(all_file_fp)):
            print(file_fp)
            for line in open(file_fp, 'r', encoding='utf-8'):
                line = line.strip().replace(' ', '').split('\t')
                if len(line) < 2:
                    error_line_num += 1
                    continue
                trg, src = line[0], line[1]
                all_trg.append(trg)
                all_src.append(src)

            if self.max_dataset_len is not None:
                if len(all_src) > self.max_dataset_len:
                    all_src = all_src[0:self.max_dataset_len]
                    all_trg = all_trg[0:self.max_dataset_len]
                    return all_src, all_trg
            gc.collect()
        print('error raw data num:', error_line_num)
        return all_src, all_trg

    def gen_pos_data(self, all_trgs):
        origin_neg_sample_num = len(all_trgs)
        pos_nums = int(self.pos_sample_gen_ratio * origin_neg_sample_num)
        if pos_nums < 1:
            logger.info('Pseudo pos data num is 0 ..')
            return []
        pos_samples = random.sample(all_trgs, pos_nums)
        return pos_samples

    @staticmethod
    def match_replace_idx(src_text, trg_text):
        """返回需要删除的索引范围, 叠字叠词删除靠后的索引

        Args:
            src ([type]): [description]
            trg ([type]): [description]

        Returns:
            [type]: [(1,2), (5,7)]
        """
        replace_range_list = []
        for diff in SequenceMatcher(None, src_text, trg_text).get_opcodes():
            tag, i1, i2, j1, j2 = diff
            if tag == 'replace':
                replace_range_list.append((i1, i2))
        return replace_range_list

    def parse_line_data_and_label(self, src, trg):
        """[summary]

        Args:
            src ([type]): text
            redundant_marks ([type]): [(1,2), (5,6)]

        Returns:
            [type]: [description]
        """

        src, trg = replace_punc_for_bert(
            src)[:self.max_seq_len - 2], replace_punc_for_bert(trg)[:self.max_seq_len - 2]
        # 对英文字级别打token，防止单词之类的#号bug
        inputs = self.list_tokenizer(src,
                                     max_len=self.max_seq_len)
        redundant_range_list = self.match_replace_idx(src, trg)
        redundant_idx_list = [
            j for i in redundant_range_list for j in range(i[0], i[1])]
        # i-1 是因为前面有cls
        d_tags = [self.dtag2id['$REPLACE'] if i-1 in redundant_idx_list else self.dtag2id['$KEEP'] for i,
                  _ in enumerate(inputs['input_ids'])]
        inputs['d_tags'] = d_tags
        return inputs

    def list_tokenizer(self, sentence_list, max_len):
        single_flag = 0
        if isinstance(sentence_list, str):
            single_flag = 1
            sentence_list = [sentence_list]

        sentence_list_limit = [sent[:max_len - 2] for sent in sentence_list]
        token_list = [[self.tokenizer.vocab.get(wd, self.tokenizer.vocab['[UNK]']) for wd in sent]
                      for sent in sentence_list_limit]
        token_list_format = [[self.tokenizer.vocab["[CLS]"]] + sent +
                             [self.tokenizer.vocab["[SEP]"]] for sent in token_list]
        sentence_list_pad = [sent + [self.tokenizer.vocab["[PAD]"]]
                             * (max_len - len(sent)) for sent in token_list_format]
        attention_mask = [
            [1] * len(sent) + [0] * (max_len - len(sent)) for sent in token_list_format

        ]
        seg_idx = [[0] * max_len for i in range(len(sentence_list_pad))]
        if single_flag == 1:
            return {
                "input_ids": sentence_list_pad[0],
                "token_type_ids": seg_idx[0],
                "attention_mask": attention_mask[0]
            }
        else:
            return {
                "input_ids": sentence_list_pad,
                "token_type_ids": seg_idx,
                "attention_mask": attention_mask
            }

    def write_data_to_file(self, fp='example_data/train/addition.txt'):

        with open(fp, 'w', encoding='utf8') as f:
            # 迭代出來的新數據添加原始文本
            [
                f.write(' '.join(map(lambda x: 'SEPL|||SEPR'.join(x), item)) +
                        '\n') for item in self.data_segments_list
            ]
    '这里有一些错字','这里又一些随时'
    def save_ds_pkl(self):
        # 数据序列化保存
        f = open(self.ds_pkl_fp, 'wb')
        logger.info('dataset pkl saved at {}'.format(self.ds_pkl_fp))
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def load_ds_pkl(ds_pkl_fp):
        f = open(ds_pkl_fp, 'rb')
        r = pickle.load(f)
        logger.info('dataset pkl loaded from {}'.format(ds_pkl_fp))
        f.close()
        return r


if __name__ == '__main__':
    data_dir_list = ['example_data/redundant']
    tokenizer_path = './pretrained_model/electra_base_cn_discriminator'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    d = DatasetDetectCsc(data_dir_list, tokenizer,
                         ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                         max_seq_len=128, max_dataset_len=10)
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=10)
    # d.write_data_to_file()
    for i in dataset:
        print(i)
