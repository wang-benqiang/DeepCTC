import random
from copy import deepcopy

import torch
from LAC import LAC
from src import logger
from src.deeplearning.dataset.lm_db.db_manager import CtcDBManager
from src.deeplearning.dataset.lm_db.yaoge_lmdb import TrainDataLmdb
from src.utils.data_helper import include_cn, replace_punc_for_bert_keep_space
from torch.utils.data import Dataset
from transformers import BertTokenizer



class MlmDataset(Dataset):
    "csc 包含检测和纠正任务"

    def __init__(self,
                 tokenizer: BertTokenizer,
                 max_seq_len,
                 max_dataset_len=None,
                 db_dir='src/deeplearning/dataset/lm_db/db/csc',
                 enable_random_mask=False
                 ):
        """

        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(MlmDataset, self).__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.max_dataset_len = max_dataset_len
        if max_dataset_len is not None:
            logger.info("train max_dataset_len:{}".format(max_dataset_len))
        self.db_dir = db_dir
        self._enable_random_mask = enable_random_mask
        
        self._mask_id = tokenizer.vocab['[MASK]']
        self._unk_id = tokenizer.vocab['[UNK]']
        self._cls_id = tokenizer.vocab['[CLS]']
        self._sep_id = tokenizer.vocab['[SEP]']
        self._loss_ignore_id = -100

        self.lac = LAC(mode='seg')
        if 'yaoge' in db_dir:
            self.db_manager = TrainDataLmdb(lmdb_dir=db_dir)
        else:
            self.db_manager = CtcDBManager(lmdb_dir=db_dir)

        logger.info('db:{}, db len: {}'.format(
            self.db_dir, len(self.db_manager)))

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
        if self.max_dataset_len is not None and self.max_dataset_len < len(self.db_manager):
            return self.max_dataset_len
        return len(self.db_manager)

    def get_random_mask_ctags(self, trg_text):
        "传入一个正确的文本，生成打mask后的标签, 这里生成的标签不带cls sep"
        enable_word_mask = random.choice([True, True, False, False, False])
        text_len = len(trg_text)
        if enable_word_mask and text_len > 10:
            trg_seg_list = self.lac.run(trg_text)
            # 可能分词后candidate_seg_dict为空，因为每个词中都可能包含非中文然后被过滤掉了，
            candidate_seg_dict = {idx: s for idx, s in enumerate(
                trg_seg_list) if all([include_cn(c) for c in s])}
            # 这里报错
            select_word_idx = random.choice(
                list(candidate_seg_dict.keys()))
            # word level mask
            # 只计算mask位置的loss
            c_tags = [self._loss_ignore_id  
                      if idx_seg != select_word_idx
                      else self.tokenizer.vocab.get(c, self._unk_id)
                      for idx_seg, seg in enumerate(trg_seg_list)
                      for c in seg]
        else:
            # char level mask
            select_char_idx = random.choice(range(text_len))
            c_tags = [self._loss_ignore_id
                      if idx != select_char_idx
                      else self.tokenizer.vocab.get(c, self._unk_id)
                      for idx, c in enumerate(trg_text)]            
        return c_tags

    def parse_line_data_and_label(self, src, trg):
        if len(src) != len(trg):
            src = trg
        src, trg = replace_punc_for_bert_keep_space(
            src)[:self.max_seq_len - 2], replace_punc_for_bert_keep_space(trg)[:self.max_seq_len - 2]
        inputs = self.list_tokenizer(src,
                                     max_len=self.max_seq_len)
        
        try:
            if self._enable_random_mask:
                "自动打mask"
                c_tags = self.get_random_mask_ctags(trg)
            else:
                "用错别字的数据打mask, 只对mask进行loss计算"
                 # '如果有正样本，因为计算loss找不到mask, 训练会出错，必须给负样本'
                if src != trg:
                    c_tags = [self.tokenizer.vocab.get(s_t[1], self._mask_id)
                            if s_t[0] != s_t[1] else self._loss_ignore_id
                            for s_t in zip(src, trg)]
                else:
                    c_tags = [self._loss_ignore_id] * len(src)
                    random_idx = random.choice(range(len(src)))
                    c_tags[random_idx] = self.tokenizer.vocab.get(src[random_idx], self._mask_id)
                
                    
        except Exception as e:
            "用错别字的数据打mask, 只对mask进行loss计算"
            logger.exception(e)
            logger.error("error text:{}".format(trg))
            c_tags = [self._loss_ignore_id] * len(src)
            c_tags[random.choice(range(len(src)))] = self._mask_id
        # 只对mask部分预测
        c_tags = [self._loss_ignore_id] + c_tags + [self._loss_ignore_id] + \
            [self._loss_ignore_id]*(self.max_seq_len - len(c_tags) - 2)
            
            
        for i, c_tag in enumerate(c_tags):
            if c_tag != self._loss_ignore_id:
                inputs['input_ids'][i] = self._mask_id
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
    from torch.utils.data import DataLoader
    pretrained_model_dir = 'model/extend_electra_small_csc'
    db_dir = 'data/train_data_csc_yaoge_0223/train_csc_lmdb_0207_1'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_dir)
    dataset = MlmDataset(
        tokenizer,
        max_seq_len=126,
        max_dataset_len=10000,
        db_dir=db_dir,
        enable_random_mask=True)

    data_loader = DataLoader(dataset, batch_size=1)

    for i in data_loader:
        print(i)
