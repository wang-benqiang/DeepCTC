import random
from difflib import SequenceMatcher
from types import coroutine

import torch
from src import logger
from src.deeplearning.dataset.lm_db.db_manager import CtcDBManager
from src.deeplearning.dataset.lm_db.yaoge_lmdb import TrainDataLmdb
from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
from src.utils.data_helper import include_cn, replace_punc_for_bert, inclue_punc
from torch.utils.data import Dataset
from pypinyin import pinyin, lazy_pinyin, Style
import os
from LAC import LAC
import json


class DatasetCscSeq2editSamePy(Dataset):
    ""

    def __init__(self,
                 tokenizer: CustomBertTokenizer,
                 max_seq_len,
                 ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                 confustion_set_dir='src/data/csc_confusion_set',
                 max_dataset_len=None,
                 db_dir='src/deeplearning/dataset/lm_db/db/ms',
                 enable_pretrain=True,
                 balanced_loss=False,
                 ):
        """
        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetCscSeq2editSamePy, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len
        self.confustion_set_dir = confustion_set_dir
        self.tokenizer = tokenizer
        self.id2dtag, self.dtag2id, self.id2ctag, self.ctag2id, self.id2py, self.py_tag2id = self.load_label_dict(
            ctc_label_vocab_dir)
        self.char_same_pinyin_confusion_set, self.word_same_pinyin_confusion_set = self.load_confusion_dict()
        self.dtag_num = len(self.dtag2id)

        self.db_dir = db_dir
        self.balanced_loss = balanced_loss
        self.enable_pretrain = enable_pretrain

        if 'yaoge' in db_dir:
            self.db_manager = TrainDataLmdb(lmdb_dir=db_dir)
        else:
            self.db_manager = CtcDBManager(lmdb_dir=db_dir)
        self.data_len = len(self.db_manager)
        logger.info('db:{}, db len: {}'.format(
            self.db_dir, len(self.db_manager)))

        self.lac = LAC(mode='seg')
        # 检测标签
        self._keep_d_tag_id, self._error_d_tag_id = self.dtag2id['$KEEP'], self.dtag2id['$ERROR']
        # self._keep_d_tag_id, self._replace_d_tag_id = self.dtag2id['$KEEP'], self.dtag2id['$REPLACE']
        # self._delete_d_tag_id, self._append_d_tag_id = self.dtag2id['$DELETE'], self.dtag2id['$APPEND']
        # 纠错标签
        self._keep_c_tag_id = self.ctag2id['$KEEP']
        self._delete_c_tag_id = self.ctag2id['$DELETE']
        self.replace_unk_c_tag_id = self.ctag2id['[REPLACE_UNK]']

        # voab id

        self._cls_vocab_id = self.tokenizer.vocab['[CLS]']
        self._sep_vocab_id = self.tokenizer.vocab['[SEP]']
        # loss ignore id

        self.punc_list_in_end = ['.', '。', '。', '！', '!']
        self.punc_list_in_end = self.punc_list_in_end + \
            [i*2 for i in self.punc_list_in_end]  # 标点数量乘2

        # load cn vocab id_lis
        self._cn_vocab = [char for (char, idx) in tokenizer.vocab.items() if len(
            char) == 1 and include_cn(char)]

        self._loss_ignore_id = -100

    def load_confusion_dict(self):
        char_level_confusion_fp = os.path.join(
            self.confustion_set_dir, 'same_py_char_confusion.json')
        word_level_confusion_fp = os.path.join(
            self.confustion_set_dir, 'same_py_word_confusion.json')

        char_same_pinyin_confusion_set = json.load(
            open(char_level_confusion_fp, encoding='utf8'))
        word_same_pinyin_confusion_set = json.load(
            open(word_level_confusion_fp, encoding='utf8'))

        logger.info('char_same_pinyin_confusion_set num: {}, word_same_pinyin_confusion_set:{}'.format(
            len(char_same_pinyin_confusion_set), len(word_same_pinyin_confusion_set)))
        return char_same_pinyin_confusion_set, word_same_pinyin_confusion_set

    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_2tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')
        pytag_fp = os.path.join(ctc_label_vocab_dir, 'char_pinyin_vocab.txt')

        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}

        id2py = [line.strip() for line in open(pytag_fp, encoding='utf8')]
        py_tag2id = {v: i for i, v in enumerate(id2py)}

        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id, id2py, py_tag2id

    @staticmethod
    def match_ctc_idx(src_text, trg_text):
        keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list = [], [], [], []

        r = SequenceMatcher(None, src_text, trg_text)
        diffs = r.get_opcodes()

        for diff in diffs:
            tag, i1, i2, j1, j2 = diff

            if tag == 'equal':
                keep_idx_list.extend(list(range(i1, i2)))
            elif tag == 'replace' and i2-i1 == j2-j1:
                # 如果文本中出现连续不同的错误label，会被diff库直接处理成replace操作
                # 所以暂时确保只有是错字类型再输出label
                replace_idx_list += [(i, '$REPLACE_'+trg_text[j])
                                     for i, j in zip(range(i1, i2), range(j1, j2))]

            # elif tag == 'insert' and j2-j1 == 1:
            #     missing_idx_list.append((i1-1, '$APPEND_'+trg_text[j1]))
            elif tag == 'delete':
                # 叠字叠词删除后面的
                redundant_length = i2-i1
                post_i1, post_i2 = i1+redundant_length, i2+redundant_length
                if src_text[i1:i2] == src_text[post_i1:post_i2]:
                    i1, i2 = post_i1, post_i2
                for i in range(i1, i2):
                    delete_idx_list.append(i)

        return keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list

    def __getitem__(self, item):
        src, trg = self.db_manager.get_src_trg(item)
        while not include_cn(src):
            # 如果数据中不包含文本.随机再抽一个数据
            src, trg = self.db_manager.get_src_trg(
                random.randint(0, self.data_len))

        if self.enable_pretrain:
            # 自动构造数据训练
            if len(trg)>4:
                word_level = random.choice([True, True, True, False])
            else:
                word_level = False
            if word_level:
                src = self.text_with_word_same_py_confusion(trg)
            else:
                src = self.text_with_char_same_py_confusion(trg)

        inputs = self.parse_line_data_and_label(src, trg)

        return_dict = {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
            'd_tags': torch.LongTensor(inputs['d_tags']),
            'c_tags': torch.LongTensor(inputs['c_tags']),
            'py_tags': torch.LongTensor(inputs['py_tags'])
        }

        return return_dict

    def __len__(self):
        if (self.max_dataset_len is not None and
                self.max_dataset_len < self.data_len):
            return self.max_dataset_len
        return self.data_len

    def convert_ids_to_ctags(self, ctag_id_list):
        return [self.id2ctag[i] if i != self._loss_ignore_id else self._loss_ignore_id for i in ctag_id_list]

    def convert_ids_to_dtags(self, dtag_id_list):
        return [self.id2dtag[i] if i != self._loss_ignore_id else self._loss_ignore_id for i in dtag_id_list]

    def parse_line_data_and_label(self, src, trg):
        """[summary]

        Args:
            src ([type]): text
            redundant_marks ([type]): [(1,2), (5,6)]

        Returns:
            [type]: [description]
        """

        if src and len(src) < 3:
            src = trg

        # 随机在句子末尾增加或删除标点，增加鲁棒性
        punc_action = random.choice([True, False])

        if punc_action and len(src) > 3:
            if inclue_punc(src[-1]):
                src, trg = src[:-1], trg[:-1]
            else:
                # 随机选一个标点加到末尾
                random_punc = random.choice(self.punc_list_in_end)
                src, trg = src+random_punc, trg+random_punc

        src, trg = replace_punc_for_bert(
            src)[:self.max_seq_len - 2], replace_punc_for_bert(trg)[:self.max_seq_len - 2]
        # 对英文字级别打token，防止单词之类的#号bug
        inputs = self.tokenizer(src,
                                max_len=self.max_seq_len)
        # 取单个value
        for k, v in inputs.items():
            inputs[k] = v[0]
        keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list = self.match_ctc_idx(
            src, trg)
        # i-1 是因为前面有cls

        # --- 计算loss ---
        src_len = len(src)
        ignore_loss_seq_len = self.max_seq_len-(src_len+1)  # sep and pad
        # 先默认给keep，会面对有错误标签的进行修改
        d_tags = [self._loss_ignore_id] + [self._keep_d_tag_id if not inclue_punc(
            c) else self._loss_ignore_id for c in src] + [self._loss_ignore_id] * ignore_loss_seq_len
        c_tags = [self._loss_ignore_id] + [self._keep_c_tag_id if not inclue_punc(
            c) else self._loss_ignore_id for c in src] + [self._loss_ignore_id] * ignore_loss_seq_len

        py_tags = self.text2pinyinids(trg)

        for (replace_idx, replace_char) in replace_idx_list:
            # +1 是因为input id的第一个字是cls
            d_tags[replace_idx+1] = self._error_d_tag_id
            c_tags[replace_idx +
                   1] = self.ctag2id.get(replace_char, self.replace_unk_c_tag_id)

        for delete_idx in delete_idx_list:
            d_tags[delete_idx+1] = self._error_d_tag_id
            c_tags[delete_idx+1] = self._delete_c_tag_id

        for (miss_idx, miss_char) in missing_idx_list:
            d_tags[miss_idx + 1] = self._error_d_tag_id
            c_tags[miss_idx +
                   1] = self.ctag2id.get(miss_char, self.append_unk_c_tag_id)

        inputs['d_tags'] = d_tags
        inputs['c_tags'] = c_tags
        inputs['py_tags'] = py_tags
        return inputs

    def text2pinyinids(self, text):
        return self.pinyins2ids(self.get_text_pinyin_list(text))

    def get_text_pinyin_list(self, text):

        pinyins = [lazy_pinyin(char, v_to_u=True)[0] if include_cn(
            char) else self._loss_ignore_id for char in text]

        return pinyins

    def pinyins2ids(self, pinyins):
        pinyin_ids = [self._loss_ignore_id] + [self.py_tag2id.get(
            pinyin, self._loss_ignore_id) for pinyin in pinyins] + [self._loss_ignore_id] * (self.max_seq_len-len(pinyins)-1)
        return pinyin_ids

    def ids2pinyins(self, py_ids):
        pinyin = [self.id2py[py_id] if py_id >=
                  0 else py_id for py_id in py_ids]
        return pinyin

    def text_with_chr_confusion(self, correct_text):

        error_text = list(correct_text)

        candidate_idx = [idx for idx, char in enumerate(
            correct_text) if include_cn(char)]

        candidate_idx_len = len(candidate_idx)

        error_num = max(
            0, self.random_error_num_by_length_stragey(candidate_idx_len))

        select_idx_li = random.sample(candidate_idx, error_num)

        for select_idx in select_idx_li:
            error_text[select_idx] = random.choice(self._cn_vocab)

        return ''.join(error_text)

    def text_with_word_same_py_confusion(self, correct_text):

        text_words = self.lac.run(correct_text)
        candidate_dict = {idx: word for idx, word in enumerate(text_words) if all(
            [include_cn(char) for char in word]) and len(word) <= 4}

        candidate_idx_len = len(candidate_dict)

        error_num = max(
            0, self.random_error_num_by_length_stragey_word_level(candidate_idx_len))

        select_idx_li = random.sample(candidate_dict.keys(), error_num)

        for select_idx in select_idx_li:
            text_words[select_idx] = self.get_word_same_py_confusion(
                text_words[select_idx])

        return ''.join(text_words)

    def text_with_char_same_py_confusion(self, correct_text):

        text_chars = list(correct_text)
        candidate_dict = {idx: char for idx, char in enumerate(
            text_chars) if all([include_cn(char) for char in text_chars])}

        candidate_idx_len = len(candidate_dict)

        error_num = max(
            0, self.random_error_num_by_length_stragey_char_level(candidate_idx_len))

        select_idx_li = random.sample(candidate_dict.keys(), error_num)

        for select_idx in select_idx_li:
            text_chars[select_idx] = self.get_char_same_py_confusion(
                text_chars[select_idx])

        return ''.join(text_chars)

    def get_char_same_py_confusion(self, char):
        char_py = lazy_pinyin(char, v_to_u=True)[0]
        # 否则从混淆字中找
        confusion_char = random.choice(
            self.char_same_pinyin_confusion_set.get(char_py, char))[0]

        return confusion_char

    def get_word_same_py_confusion(self, word):
        word_py_list = lazy_pinyin(word, v_to_u=True)
        word_py = ''.join(word_py_list)
        if word_py in self.word_same_pinyin_confusion_set:
            # 如果在混淆词中
            confusion_word = random.choice(
                self.word_same_pinyin_confusion_set[word_py])[0]
        else:
            # 否则从混淆字中找
            confusion_word = ''.join([random.choice(
                self.char_same_pinyin_confusion_set.get(char_py, word[idx]))[0] for idx, char_py in enumerate(word_py_list)])

        # 有可能找的是原词/字，再字粒度重新找一次
        if confusion_word == word:
            confusion_word = ''.join([random.choice(
                self.char_same_pinyin_confusion_set.get(char_py, word[idx]))[0] for idx, char_py in enumerate(word_py_list)])

        return confusion_word

    def random_error_num_by_length_stragey_word_level(self, candidate_idx_len):

        stragey = {
            range(0, 3): [0],
            range(3, 4): [0, 1],
            range(4, 7): [0, 1, 1, 1],
            range(7, 12): [1, 1, 1, 1, 1, 2],
            range(12, 32): [1, 1, 1, 1, 1, 2, 2],
            range(32, 64): [1, 1, 1, 1, 1, 2, 2, 2, 3],
            range(64, 129): [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4],
        }

        for len_range, error_num_li in stragey.items():
            if candidate_idx_len in len_range:
                return random.choice(error_num_li)

        return 0

    def random_error_num_by_length_stragey_char_level(self, candidate_idx_len):
        stragey = {

            range(0, 2): [0],
            range(2, 5): [0, 1, 1],
            range(5, 12): [1, 1, 1, 1, 2, 2],
            range(12, 32): [1, 1, 1, 2, 2, 2, 3],
            range(32, 64): [1, 1, 1, 2, 2, 2, 3, 3, 4],
            range(64, 129): [1, 1, 1,  2, 2, 2, 2, 3, 3, 3, 4, 5],
        }
        for len_range, error_num_li in stragey.items():
            if candidate_idx_len in len_range:
                return random.choice(error_num_li)
        return 0


if __name__ == '__main__':

    tokenizer_path = 'model/extend_electra_base'
    tokenizer = CustomBertTokenizer.from_pretrained(tokenizer_path)
    d = DatasetCscSeq2editSamePy(tokenizer,
                                 max_seq_len=128,
                                 ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                                 db_dir='db/realise_train_ft',
                                 balanced_loss=False,
                                 enable_pretrain=True)
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=158)
    for i in dataset:
        print(i)

    src_text = '可老爸还是无动于束'
    trg_text = '可老爸还是无动于衷'

    r = d.match_ctc_idx(src_text, trg_text)

    r = d.parse_line_data_and_label(src_text, trg_text)
    x = iter(d)
    r = next(x)
    print(r)
