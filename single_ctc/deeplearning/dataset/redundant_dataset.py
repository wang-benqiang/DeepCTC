from difflib import SequenceMatcher

import torch
from src import logger
from src.deeplearning.dataset.lm_db.db_manager import CtcDBManager
from src.utils.data_helper import replace_punc_for_bert
from src.deeplearning.dataset.lm_db.yaoge_lmdb import TrainDataLmdb
from torch.utils.data import Dataset
from transformers import BertTokenizer


class DatasetRedundant(Dataset):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 max_seq_len,
                 ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                 max_dataset_len=None,
                 db_dir='src/deeplearning/dataset/lm_db/db/rdt'
                 ):
        """
        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetRedundant, self).__init__()
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
            
        logger.info('db:{}, db len: {}'.format(
        self.db_dir, len(self.db_manager)))


    def load_label_dict(self, ctc_label_vocab_dir):

        dtag_fp = '{}/{}'.format(ctc_label_vocab_dir, 'redundant_tags.txt')

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
        if self.max_dataset_len is not None and self.max_dataset_len < self.data_len:
            return self.max_dataset_len
        return self.data_len

    @staticmethod
    def match_redundant_idx(src_text, trg_text):
        """返回需要删除的索引范围, 叠字叠词删除靠后的索引

        Args:
            src ([type]): [description]
            trg ([type]): [description]

        Returns:
            [type]: [(1,2), (5,7)]
        """
        r = SequenceMatcher(None, src_text, trg_text)
        diffs = r.get_opcodes()
        redundant_range_list = []
        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag == 'delete':
                redundant_length = i2-i1
                post_i1, post_i2 = i1+redundant_length, i2+redundant_length
                if src_text[i1:i2] == src_text[post_i1:post_i2]:
                    redundant_range_list.append((post_i1, post_i2))
                else:
                    redundant_range_list.append((i1, i2))
        return redundant_range_list

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
        redundant_range_list = self.match_redundant_idx(src, trg)
        redundant_idx_list = [
            j for i in redundant_range_list for j in range(i[0], i[1])]
        # i-1 是因为前面有cls
        d_tags = [self.dtag2id['$DELETE'] if i-1 in redundant_idx_list else self.dtag2id['$KEEP'] for i,
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


if __name__ == '__main__':
    data_dir_list = ['example_data/redundant']
    tokenizer_path = './pretrained_model/electra_base_cn_discriminator'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    d = DatasetRedundant(data_dir_list, tokenizer,
                         ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                         max_seq_len=128, max_dataset_len=10)
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=10)
    # d.write_data_to_file()
    for i in dataset:
        print(i)
