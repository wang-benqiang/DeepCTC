import random
from difflib import SequenceMatcher

import torch
from src import logger
from src.deeplearning.dataset.lm_db.db_manager import CtcDBManager
from src.deeplearning.dataset.lm_db.yaoge_lmdb import TrainDataLmdb
from src.utils.data_helper import replace_punc_for_bert, inclue_punc
from torch.utils.data import Dataset
from transformers import BertTokenizer


class DatasetGec(Dataset):
    "多字少字Dataset"

    def __init__(self,
                 tokenizer: BertTokenizer,
                 max_seq_len,
                 ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                 max_dataset_len=None,
                 db_dir='src/deeplearning/dataset/lm_db/db/ms'
                 ):
        """
        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetGec, self).__init__()
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
        logger.info('db:{}, db len: {}'.format(
            self.db_dir, len(self.db_manager)))

        # 检测标签 和 纠正标签
        self._loss_ignore_id = -100
        self.right_d_tag_id, self.error_d_tag_id = self.dtag2id['$RIGHT'], self.dtag2id['$ERROR']
        self.keep_c_tag_id = self.tokenizer.convert_tokens_to_ids('$KEEP')
        self.delete_c_tag_id = self.tokenizer.convert_tokens_to_ids('$DELETE')

    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = '{}/{}'.format(ctc_label_vocab_dir, 'gec_tags.txt')
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id,

    @staticmethod
    def match_gec_idx(src_text, trg_text):
        delete_idx_list, missing_idx_list = [], []
        if src_text != trg_text:
            r = SequenceMatcher(None, src_text, trg_text)
            diffs = r.get_opcodes()

            for diff in diffs:
                tag, i1, i2, j1, j2 = diff
                if tag == 'insert' and j2-j1 == 1:
                    missing_idx_list.append((i1-1, trg_text[j1]))
                elif tag == 'delete':
                    # 叠字叠词删除后面的
                    redundant_length = i2-i1
                    post_i1, post_i2 = i1+redundant_length, i2+redundant_length
                    if src_text[i1:i2] == src_text[post_i1:post_i2]:
                        i1, i2 = post_i1, post_i2
                    for i in range(i1, i2):
                        delete_idx_list.append(i)

        return delete_idx_list, missing_idx_list

    def __getitem__(self, item):
        src, trg = self.db_manager.get_src_trg(item)
        inputs = self.parse_line_data_and_label(src, trg)

        return {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
            'd_tags': torch.LongTensor(inputs['d_tags']),
            'c_tags': torch.LongTensor(inputs['c_tags'])
        }

    def __len__(self):
        if (self.max_dataset_len is not None and
                self.max_dataset_len < self.data_len):
            return self.max_dataset_len
        return self.data_len

    def parse_line_data_and_label(self, src, trg):
        """[summary]

        Args:
            src ([type]): text
            redundant_marks ([type]): [(1,2), (5,6)]

        Returns:
            [type]: [description]
        """
        if len(trg) < 3 or abs(len(src)-len(trg))>6:
            src = trg
            
        # 随机在句子末尾增加或删除标点，增加鲁棒性
        punc_action = random.choice([True, False])
        
        if punc_action and len(src)>3:
            if inclue_punc(src[-1]):
                src, trg = src[:-1], trg[:-1]
            else:
                # 随机选一个标点加到末尾
                random_punc = random.choice(['.','。','。','！', '!'])
                src, trg = src+random_punc, trg+random_punc
                
        src, trg = replace_punc_for_bert(
            '始'+src)[:self.max_seq_len - 2], replace_punc_for_bert('始'+trg)[:self.max_seq_len - 2]
        # 对英文字级别打token，防止单词之类的#号bug
        inputs = self.list_tokenizer(src,
                                     max_len=self.max_seq_len)
        inputs['input_ids'][1] = 1  # 把 始 换成 [unused1]
        delete_idx_list, missing_idx_list = self.match_gec_idx(src, trg)
        # i-1 是因为前面有cls

        src_len = len(src)
        ignore_loss_seq_len= self.max_seq_len-(src_len+1)  # sep and pad

        d_tags = [self._loss_ignore_id] + [self.right_d_tag_id] * src_len + [self._loss_ignore_id] * ignore_loss_seq_len
        c_tags = [self._loss_ignore_id] + [self.keep_c_tag_id] * src_len + [self._loss_ignore_id] * ignore_loss_seq_len
        for delete_idx in delete_idx_list:
            # +1 是因为input id的第一个字是cls
            d_tags[delete_idx+1] = self.error_d_tag_id
            c_tags[delete_idx+1] = self.delete_c_tag_id

        for (miss_idx, miss_char) in missing_idx_list:
            d_tags[miss_idx + 1] = self.error_d_tag_id
            c_tags[miss_idx +
                   1] = self.tokenizer.convert_tokens_to_ids(miss_char)
            
        inputs['d_tags'] = d_tags
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

    tokenizer_path = 'model/miduCTC_v3.7.0_csc3model/gec'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    d = DatasetGec(tokenizer,
                   max_seq_len=122,
                   ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                   db_dir='db/gec_ft_test')
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=10)
    # # d.write_data_to_file()
    for i in dataset:
        print(i)

    # src_text = '这是一个这是一个句子,是的我知道这件'
    # trg_text = '这是一个句子,是的我知道这件事'
    # r = d.parse_line_data_and_label(src_text, trg_text)
    # x = iter(d)
    # r = next(x)
    # print(r)
