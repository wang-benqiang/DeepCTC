import gc
import os
import pickle
import random
import time

import torch
from rich.progress import track
from src import logger
from src.deeplearning.dataset.lm_db.db_manager import CtcDBManager
from src.utils.data_helper import replace_punc_for_bert
from src.deeplearning.dataset.lm_db.yaoge_lmdb import TrainDataLmdb
from src.utils.realise.pinyin_util import Pinyin2
from torch.utils.data import Dataset
from transformers import BertTokenizer
from src.deeplearning.modeling.modeling_realise import SpellBertPho2ResArch3


class CscQuantDataset(Dataset):
    def __init__(self,
                 data_dir_list,
                 tokenizer: BertTokenizer,
                 max_seq_len,
                 max_dataset_len=None,
                 max_len_char_pinyin=7,
                 pos_sample_gen_ratio=0.1,
                 db_dir='src/deeplearning/dataset/lm_db/db/csc',
                 ):
        """

        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param keep_one_append:多个操作型label保留一个
        """
        super(CscQuantDataset, self).__init__()
        self.data_dir_list = data_dir_list
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len
        self.max_len_char_pinyin = max_len_char_pinyin
        self.tokenizer = tokenizer
        self.pos_sample_gen_ratio = pos_sample_gen_ratio
        self.pho2_convertor = Pinyin2()
        
        self.db_dir = db_dir
      
        if 'yaoge' in db_dir:
            self.db_manager = TrainDataLmdb(lmdb_dir=db_dir)
        else:
            self.db_manager = CtcDBManager(lmdb_dir=db_dir)
       

            
            
        self.data_len = len(self.db_manager)
    
    

    def __getitem__(self, item):
        
        src, trg = self.db_manager.get_src_trg(item)
        inputs = self.parse_line_data_and_label(src, trg)

        return_dict = {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
            'c_tags': torch.LongTensor(inputs['c_tags']),
        }

        return return_dict

    def __len__(self):
        if self.max_dataset_len is not None and self.max_dataset_len<self.data_len:
            return self.max_dataset_len
        return self.data_len


  

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
            print(data_dir)
            print(current_dir_files)
            current_dir_file_fp = [
                '{}/{}'.format(data_dir, f_name) for f_name in current_dir_files
            ]
            all_file_fp.extend(current_dir_file_fp)

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

        all_src, all_trg = [], []
        print('loading {} files:{}'.format(len(all_file_fp), str(all_file_fp)))
        for file_fp in track(all_file_fp,
                             description='Processing raw data...',
                             total=len(all_file_fp)):

            for line in open(file_fp, 'r', encoding='utf-8'):
                line = line.strip().replace(' ', '').split('\t')
                try:
                    if len(line[0]) == len(line[1]):
                        all_trg.append(line[0])
                        all_src.append(line[1])
                except:
                    print(line)
            if self.max_dataset_len is not None:
                if len(all_src) > self.max_dataset_len:
                    all_src = all_src[0:self.max_dataset_len]
                    all_trg = all_trg[0:self.max_dataset_len]
                    return all_src, all_trg
            gc.collect()
        return all_src, all_trg



    def gen_pos_data(self, all_trgs):
        origin_neg_sample_num = len(all_trgs)
        logger.info('pos_sample_gen_ratio: {}'.format(self.pos_sample_gen_ratio))
        pos_nums = int(self.pos_sample_gen_ratio * origin_neg_sample_num)
        if pos_nums < 1:
            logger.info('Pseudo pos data num is 0 ..')
            return []
        pos_samples = random.sample(all_trgs, pos_nums)
        return pos_samples

    def parse_line_data_and_label(self, src, trg):
        # [[['$START', '$KEEP'], ['明', '$KEEP']], ...]
        # 保证传入的数据里没有空格如：[' ', '$KEEP']

        src, trg = replace_punc_for_bert(
            src)[:self.max_seq_len - 2], replace_punc_for_bert(trg)[:self.max_seq_len - 2]
        # 对英文字级别打token，防止单词之类的#号bug

        inputs = self.list_tokenizer(src,
                                     max_len=self.max_seq_len)
        # あ
        c_tags = [self.tokenizer.vocab.get(trg_char, self.tokenizer.vocab['あ']) 
                  if src_char!=trg_char else self.tokenizer.vocab['あ']
                  for src_char, trg_char in zip(src, trg)]
        c_tags = [-1] + c_tags + [-1] * (self.max_seq_len - len(c_tags) - 1)
        inputs['c_tags'] = c_tags
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



if __name__ == '__main__':
    data_dir_list = ['example_data/csc', '']
    tokenizer_path = './pretrained_model/electra_base_cn_discriminator'

    # tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    # d = RealiseRawDataset(data_dir_list, tokenizer,
    #                       max_seq_len=128,
    #                       max_dataset_len=2,
    #                       task_types=['correct', 'detect', 'pinyin'],
    #                       db_dir='src/deeplearning/dataset/lm_db/db/csc',
    #                       pos_sample_gen_ratio=0.5)
    # dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=10)
    # # d.write_data_to_file()
    # for i in dataset:
    #     i['pho_idx'] = i['pho_idx'].view(-1, 7)
    #     i['pho_lens'] = i['pho_lens'].flatten().tolist()
    
    model_condig = SpellBertPho2ResArch3.config_class.from_pretrained('model/extend_realise')
    print(model_condig)