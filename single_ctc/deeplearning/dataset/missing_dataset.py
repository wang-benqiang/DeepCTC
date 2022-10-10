import gc
import os
import random
from difflib import SequenceMatcher
from src.deeplearning.dataset.lm_db.yaoge_lmdb import TrainDataLmdb

import torch
from rich.progress import track
from src import logger
from src.deeplearning.dataset.lm_db.db_manager import CtcDBManager
from src.utils.data_helper import replace_punc_for_bert
from torch.utils.data import Dataset
from transformers import BertTokenizer


class DatasetMissing(Dataset):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 max_seq_len,
                 ctc_label_vocab_dir,
                 max_dataset_len=None,
                 db_dir='src/deeplearning/dataset/lm_db/db/ms',
                 ):
        """
        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetMissing, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len
        self.tokenizer = tokenizer
        self.id2dtag, self.dtag2id = self.load_label_dict(
            ctc_label_vocab_dir)

        self.dtag_num = len(self.dtag2id)

        self.db_dir = db_dir
        if 'yaoge' in db_dir:
            self.db_manager = TrainDataLmdb(lmdb_dir=db_dir)
        else:
            self.db_manager = CtcDBManager(lmdb_dir=db_dir)

        self.data_len = len(self.db_manager)

        gc.collect()

    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = '{}/{}'.format(ctc_label_vocab_dir, 'missing_tags.txt')
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id,

    @staticmethod
    def gen_missing1char_src_trg(src, trg):
        """返回只少一个字的src和trg

        Args:
            src ([type]): '你是一好人,我这件事情'
            trg ([type]): '你是一个好人,我知道这件事事情'

        Returns:
            [type]: '你是一好人,我这件事情', '你是一个好人,我知这件事事情']
        """
        if src == trg:
            return None
        r = SequenceMatcher(None, src, trg)
        diffs = r.get_opcodes()
        _trg = list(src)
        insert_flag = False  # 只生成带有insert行为的数据
        insert_count = 0
        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag == 'insert':
                insert_flag = True
                _trg.insert(i1 + insert_count, trg[j1])
                insert_count += 1
        if insert_flag:
            return src, ''.join(_trg)
        else:
            return None

    @staticmethod
    def gen_src_trg_with_iteration(src_str, trg_str):
        """[迭代生成连续少多个字的情况]

        Args:
            src_str ([type]): '你是一好人,我这件事情'
            trg_str ([type]): '你是一个好人,我知道这件事事情'

        Returns:
            src_texts: ['你是一好人,我这件事情', '你是一个好人,我知这件事事情']
            trg_texts : ['你是一个好人,我知这件事事情', '你是一个好人,我知道这件事事情']
        """

        src_texts, trg_texts = [], []
        if src_str == trg_str:
            src_texts.append(src_str)
            trg_texts.append(trg_str)
            return src_texts, trg_texts
        r = DatasetMissing.gen_missing1char_src_trg(src_str, trg_str)
        while r is not None:
            src_texts.append(r[0])
            trg_texts.append(r[1])
            r = DatasetMissing.gen_missing1char_src_trg(r[1], trg_str)
        return src_texts, trg_texts

    @staticmethod
    def match_missing_idx(src_text, trg_text):
        r = SequenceMatcher(None, src_text, trg_text)
        diffs = r.get_opcodes()
        missing_idx_list = []
        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag == 'insert':
                missing_idx_list.append(i1-1)
        return missing_idx_list

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
        if self.max_dataset_len is not None and self.max_dataset_len < self.data_len:
            return self.max_dataset_len
        return self.data_len


    def gen_pos_data(self, all_trgs):
        origin_neg_sample_num = len(all_trgs)
        pos_nums = int(self.pos_sample_gen_ratio * origin_neg_sample_num)
        if pos_nums < 1:
            logger.info('Pseudo pos data num is 0 ..')
            return []
        pos_samples = random.sample(all_trgs, pos_nums)
        return pos_samples

    def parse_line_data_and_label(self, src, trg):
        """[summary]

        Args:
            src ([type]): text
            redundant_marks ([type]): [(1,2), (5,6)]

        Returns:
            [type]: [description]
        """

        src, trg = replace_punc_for_bert(
            '始'+src)[:self.max_seq_len - 2], replace_punc_for_bert('始'+trg)[:self.max_seq_len - 2]
        # 对英文字级别打token，防止单词之类的#号bug
        inputs = self.list_tokenizer(src,
                                     max_len=self.max_seq_len)
        inputs['input_ids'][1] = 1  # 把 始 换成 [unused1]
        missing_idx_list = self.match_missing_idx(src, trg)
        # i-1 是因为前面有cls
        d_tags = [self.dtag2id['$APPEND'] if i-1 in missing_idx_list else self.dtag2id['$KEEP'] for i,
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


if __name__ == '__main__':
    data_dir_list = ['example_data/missing']
    tokenizer_path = './pretrained_model/electra_base_cn_discriminator'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    d = DatasetMissing(data_dir_list, tokenizer,
                       ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                       max_seq_len=128, max_dataset_len=10)
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=10)
    # d.write_data_to_file()
    for i in dataset:
        print(i)
