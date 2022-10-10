import gc
import os
import random
from difflib import SequenceMatcher

import torch
from logs import logger
from src.modeling.modeling_csc_realise import ModelingCscRealise
from src.tokenizer.bert_tokenizer import CustomBertTokenizer
from torch.utils.data import Dataset
from utils.data_helper import include_cn, inclue_punc, replace_punc_for_bert
from utils.lmdb.db_manager import CtcDBManager
from utils.lmdb.yaoge_lmdb import TrainDataLmdb
from utils.pinyin_util import Pinyin2


class DatasetCscRealise(Dataset):
    def __init__(self,
                 in_model_dir,
                 tokenizer: CustomBertTokenizer,
                 max_seq_len,
                 max_len_char_pinyin=7,
                 max_dataset_len=None,
                 db_dir='db/realise_train_ft',
                 pinyin_data_dir='src/vocab',
                 only_error_loss=False,
                 ):
        """

        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetCscRealise, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len
        self.max_len_char_pinyin = max_len_char_pinyin
        self.pinyin_data_dir = pinyin_data_dir
        self.tokenizer = tokenizer
        self.only_error_loss = only_error_loss  # 只针对有错误的地方计算loss
        self.model_config = ModelingCscRealise.config_class.from_pretrained(
            in_model_dir)
        if self.model_config.add_pinyin_task:
            self.char2pinyin, self.pinyin2id, self.id2pinyin = self.load_char_pinyin_vocab()
        self.pho2_convertor = Pinyin2()
        self.db_dir = db_dir
        if 'yaoge' in db_dir:
            self.db_manager = TrainDataLmdb(lmdb_dir=db_dir)
        else:
            self.db_manager = CtcDBManager(lmdb_dir=db_dir)
        self.loss_ignore_id = -100

        self.punc_list_in_end = ['.', '。', '。', '！', '!']
        self.punc_list_in_end = self.punc_list_in_end + \
            [i*2 for i in self.punc_list_in_end]  # 标点数量乘2
        self.data_len = len(self.db_manager)
        logger.info('db:{}, db len: {}'.format(
            self.db_dir, len(self.db_manager)))

    def __getitem__(self, item):

        src, trg = self.db_manager.get_src_trg(item)
        while not include_cn(src):
            # 如果数据中不包含文本.随机再抽一个数据
            src, trg = self.db_manager.get_src_trg(
                random.randint(0, self.data_len))
        inputs = self.parse_line_data_and_label(src, trg)
        return_dict = {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'c_tags': torch.LongTensor(inputs['c_tags']),
            'pho_idx': torch.LongTensor(inputs['pho_idx']),
            'pho_lens': torch.LongTensor(inputs['pho_lens']),
        }

        if self.model_config.add_detect_task:
            return_dict['d_tags'] = torch.LongTensor(inputs['d_tags'])
        if self.model_config.add_pinyin_task:
            return_dict['pinyin_tags'] = torch.LongTensor(
                inputs['pinyin_tags'])

        return return_dict

    def __len__(self):
        if (self.max_dataset_len > 0 and self.max_dataset_len < self.data_len):
            return self.max_dataset_len

        return self.data_len

    def load_char_pinyin_vocab(self):
        fp = os.path.join(self.pinyin_data_dir, 'py_vocab.txt')
        id2pinyins = [i.strip() for i in open(fp, 'r', encoding='utf-8')]
        pinyin2id = {pinyin: idx for idx, pinyin in enumerate(id2pinyins)}

        fp = os.path.join(self.pinyin_data_dir, 'char_py.txt')
        char2pinyin = {}
        for i in open(fp, 'r', encoding='utf-8'):
            char, pinyin = i.strip().split('\t')
            char2pinyin.setdefault(char, pinyin)
        return char2pinyin, pinyin2id, id2pinyins

    def char2pinyinid(self, char):
        if not inclue_punc(char):
            pinyin = self.char2pinyin.get(char, '[UNK]')
            return self.pinyin2id.get(pinyin, self.pinyin2id['[UNK]'])
        else:
            return self.loss_ignore_id

    def parse_line_data_and_label(self, src, trg):
        # [[['$START', '$KEEP'], ['明', '$KEEP']], ...]
        # 保证传入的数据里没有空格如：[' ', '$KEEP']
        if len(src) != len(trg):
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
        for k,v in inputs.items():
            inputs[k] = v[0] # get single data
        input_chars = self.tokenizer.convert_ids_to_tokens(
            inputs['input_ids'])

        phos_idx, phos_lens = self.pho2_convertor.convert(
            input_chars, self.max_len_char_pinyin)
        """
        [[32,  0,  0,  ...,  0,  0,  0],
            [ 2, 22, 14,  ...,  0,  0,  0],
            [ 4, 29, 14,  ..., 20,  0,  0]
        ],
        [1, 3, 5]
        """
        inputs['pho_idx'] = phos_idx
        inputs['pho_lens'] = phos_lens

        if not self.only_error_loss:
            c_tags = [self.tokenizer.vocab.get(
                wd, self.tokenizer.vocab['[UNK]']) if not inclue_punc(wd) else self.loss_ignore_id for wd in trg]
            c_tags = [self.loss_ignore_id] + c_tags + \
                [self.loss_ignore_id] * (self.max_seq_len - len(c_tags) - 1)
            inputs['c_tags'] = c_tags

            if self.model_config.add_detect_task:
                # if realise include detect task
                d_tags = [self.loss_ignore_id] + [self.parse_dtag(i[0], i[1]) for i in zip(
                    src, trg)] + [self.loss_ignore_id] * (self.max_seq_len - len(trg) - 1)
                inputs['d_tags'] = d_tags

            if self.model_config.add_pinyin_task:
                # if realise include pinyin prediction task
                pinyin_tags = [self.char2pinyinid(char) for char in trg]
                pinyin_tags = [self.loss_ignore_id] + pinyin_tags + [self.loss_ignore_id] * \
                    (self.max_seq_len - len(pinyin_tags) - 1)
                inputs['pinyin_tags'] = pinyin_tags
        else:

            input_ids_len = len(inputs['input_ids'])
            diffs = SequenceMatcher(None, src, trg).get_opcodes()
            eaqul_ids, replace_id_char = [], {}
            for tag, i1, i2, j1, j2 in diffs:
                if tag == 'equal':
                    eaqul_ids.extend(list(range(i1, i2)))
                elif tag == 'replace' and j2-j1 == i2-i1:
                    for i, j in zip(range(i1, i2), range(j1, j2)):
                        replace_id_char[i] = trg[j]
            c_tags = [self.loss_ignore_id] * input_ids_len
            d_tags = [self.loss_ignore_id] * input_ids_len
            pinyin_tags = [self.loss_ignore_id] * input_ids_len
            for idx, trg_char in replace_id_char.items():
                c_tags[idx+1] = self.tokenizer.vocab.get(
                    trg_char, self.tokenizer.vocab['[UNK]'])
                d_tags[idx+1] = 1
                pinyin_tags[idx+1] = self.char2pinyinid(trg_char)
            if eaqul_ids:
                # 如果有想同的字，随机给一个计算loss
                random_pos_id = random.choice(eaqul_ids)

                c_tags[random_pos_id + 1] = self.tokenizer.vocab.get(
                    src[random_pos_id], self.tokenizer.vocab['[UNK]'])
                d_tags[random_pos_id + 1] = 0
                pinyin_tags[random_pos_id +
                            1] = self.char2pinyinid(src[random_pos_id])

            inputs['c_tags'] = c_tags
            inputs['d_tags'] = d_tags
            inputs['pinyin_tags'] = pinyin_tags

        return inputs

    def parse_dtag(self, src_char, trg_char):
        if not inclue_punc(src_char):
            if src_char != trg_char:
                return 1
            else:
                # 正确的字是0
                return 0
        else:
            return self.loss_ignore_id


if __name__ == '__main__':

    model_dir = 'model/realise_ccl2022_2022Y07M24D23H/epoch4,ith_db:0,step75,testf1_99_57%,devf1_99_62%'

    tokenizer = CustomBertTokenizer.from_pretrained(model_dir)
    d = DatasetCscRealise(
        in_model_dir=model_dir,
        tokenizer=tokenizer,
        max_seq_len=128,
        max_dataset_len=1000,
        db_dir='data/lmdb/ccl2022_finetune_dev_test'
    )
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=10)
    for i in dataset:
        print(i)

  
