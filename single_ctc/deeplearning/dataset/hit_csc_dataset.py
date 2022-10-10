import gc
import os
import pickle
import random

import torch
from rich.progress import track
from src import logger
from src.deeplearning.dataset.lm_db.db_manager import CtcDBManager
from src.utils.data_helper import replace_punc_for_bert
from src.utils.realise.pinyin_util import Pinyin2
from torch.utils.data import Dataset
from transformers import BertTokenizer


class HitCscDataset(Dataset):
    def __init__(self,
                 data_dir_list,
                 tokenizer: BertTokenizer,
                 max_seq_len,
                 max_dataset_len=None,
                 pos_sample_gen_ratio=0.5,
                 ds_pkl_fp=None,
                 db_dir='src/deeplearning/dataset/lm_db/db/csc_hit',
                 db_gb_size = 20,
                 ):
        """ hit csc 任务, 错字回归到正确的字, 正确的字统一回归到一个label:あ
        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(HitCscDataset, self).__init__()
        self.data_dir_list = data_dir_list
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len

        self.ds_pkl_fp = ds_pkl_fp
        self.tokenizer = tokenizer
        self.pos_sample_gen_ratio = pos_sample_gen_ratio
        self.pho2_convertor = Pinyin2()
        
        
        self.db_dir = db_dir
        if os.path.exists(self.db_dir):
            self.db_manager = CtcDBManager(lmdb_dir=db_dir)
        else:
            all_src, all_trg = self._read_ctc_data_from_file()
            logger.info('neg samples loaded, num: {}'.format(
                len(all_src)))
            pos_samples = self.gen_pos_data(all_trg)
            logger.info('pos samples loaded, num: {}'.format(len(pos_samples)))
            if len(pos_samples) > 0:
                all_src.extend(pos_samples), all_trg.extend(pos_samples)
            self.db_manager = CtcDBManager(lmdb_dir=db_dir)
            self.db_manager.create_db(all_src, all_trg,  db_gb_size=db_gb_size)

        self.data_len = self.db_manager.get_data_len()
        logger.info('all samples loaded, num: {}'.format(
            self.data_len))

        all_src, all_trg = [], []
        gc.collect()
        if self.ds_pkl_fp is not None:
            self.save_ds_pkl()
            logger.info('dataset pkl saved at {}'.format(self.ds_pkl_fp))

    def __getitem__(self, item):
        src, trg = self.db_manager.get_src_trg(item)
        inputs = self.parse_line_data_and_label(src, trg)
        return {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
            'c_tags': torch.LongTensor(inputs['c_tags'])
        }

    def __len__(self):
        return len(self.all_trg)

    def _read_ctc_data_from_file(self):
        """读取ctc数据

        Args:
            keep_one_append ([type]): 是否迭代生成多条数据
            chunk_num (int, optional): [description]. Defaults to 100000.

        Returns:
            list:[ [['$START', '$KEEP'], ['明', '$KEEP']], ...]
        """
        all_file_fp = []
        for data_dir in self.data_dir_list:
            current_dir_files = os.listdir(data_dir)
            current_dir_file_fp = [
                '{}/{}'.format(data_dir, f_name) for f_name in current_dir_files
            ]
            all_file_fp.extend(current_dir_file_fp)
        all_src, all_trg = [], []
        print('loading {} files:{}'.format(len(all_file_fp), str(all_file_fp)))
        for file_fp in track(all_file_fp,
                             description='Processing raw data...',
                             total=len(all_file_fp)):
            for line in open(file_fp, 'r', encoding='utf-8'):
                line = line.strip().replace(' ', '').split('\t')
                if len(line[0]) == len(line[1]):
                    all_trg.append(line[0])
                    all_src.append(line[1])

            if self.max_dataset_len is not None:
                if len(all_src) > self.max_dataset_len:
                    all_src = all_src[0:self.max_dataset_len]
                    all_trg = all_trg[0:self.max_dataset_len]
                    return all_src, all_trg
            gc.collect()
        return all_src, all_trg

    def char_label_to_correct_charid(self, char_label_list):
        """  
        Args:
            char_label_list ([type]): ['明', '$KEEP']
        Return:
            明的id
        """

        if '$REPLACE_' not in char_label_list[1]:
            char = char_label_list[0]
        else:

            char = char_label_list[1].split('_')[1]

        char_id = self.tokenizer.convert_tokens_to_ids(char)
        # 如果出现不认识的字可能会是100 unk
        return char_id

    def char_label_to_correct_detect_id(self, char_label_list):
        """ 

        Args:
            char_label_list ([type]): ['明', '$KEEP']
        Return:
            0
        """

        if '$REPLACE_' in char_label_list[1]:
            return 1  # 是replace
        return 0  # 0是$keep

    def gen_pos_data(self, all_trgs):
        origin_neg_sample_num = len(all_trgs)
        pos_nums = int(self.pos_sample_gen_ratio * origin_neg_sample_num)
        if pos_nums < 1:
            logger.info('Pseudo pos data num is 0 ..')
            return []
        pos_samples = random.sample(all_trgs, pos_nums)
        return pos_samples

    def parse_line_data_and_label(self, src, trg):
        src, trg = replace_punc_for_bert(
            src)[:self.max_seq_len - 2], replace_punc_for_bert(trg)[:self.max_seq_len - 2]
        inputs = self.list_tokenizer(src,
                                     max_len=self.max_seq_len)
        c_tags = [self.tokenizer.vocab.get(s_t[1], self.tokenizer.vocab['[UNK]'])
                  if s_t[0] != s_t[1]
                  else self.tokenizer.vocab['あ']
                  for s_t in zip(src, trg)]
        c_tags = [-1] + c_tags + [-1] * (self.max_seq_len - len(c_tags) - 1)
        inputs['c_tags'] = c_tags
        inputs['raw_data'] = ' '.join(
            self.tokenizer.convert_ids_to_tokens(inputs['input_ids']))
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
    data_dir_list = ['example_data/csc']
    tokenizer_path = './pretrained_model/electra_base_cn_discriminator'

    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    d = HitCscDataset(data_dir_list,
                      tokenizer,
                      max_seq_len=12,
                      max_dataset_len=2,
                      db_dir='src/deeplearning/dataset/lm_db/db/csc_hit',
                      pos_sample_gen_ratio=0.5)
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=10)
    # d.write_data_to_file()
    for i in dataset:
        print(i)
