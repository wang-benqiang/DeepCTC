import random
from difflib import SequenceMatcher

import torch
from src import logger
from src.deeplearning.dataset.lm_db.db_manager import CtcDBManager
from src.deeplearning.dataset.lm_db.yaoge_lmdb import TrainDataLmdb
from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
from src.utils.data_helper import include_cn, replace_punc_for_bert, inclue_punc
from torch.utils.data import Dataset

import os


class DatasetCscLm(Dataset):

    def __init__(self,
                 tokenizer: CustomBertTokenizer,
                 max_seq_len,
                 ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                 max_dataset_len=None,
                 db_dir='src/deeplearning/dataset/lm_db/db/ms',
                 enable_pretrain=True,
                 ):
        """
        char level bert for error detection
        """

        super(DatasetCscLm, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len
        self.tokenizer = tokenizer
        self.id2dtag, self.dtag2id = self.load_label_dict(
            ctc_label_vocab_dir)

        self.dtag_num = len(self.dtag2id)

        self.db_dir = db_dir
   
        
        self.enable_pretrain = enable_pretrain

        if 'yaoge' in db_dir:
            self.db_manager = TrainDataLmdb(lmdb_dir=db_dir)
        else:
            self.db_manager = CtcDBManager(lmdb_dir=db_dir)
        self.data_len = len(self.db_manager)
        logger.info('db:{}, db len: {}'.format(
            self.db_dir, len(self.db_manager)))

        # 检测标签
        self._keep_d_tag_id, self._error_d_tag_id = self.dtag2id['$KEEP'], self.dtag2id['$ERROR']
    
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

    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_2tags.txt')
       

        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

 
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id

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

        return keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list

    def __getitem__(self, item):
        src, trg = self.db_manager.get_src_trg(item)
        while not include_cn(src):
            # 如果数据中不包含文本.随机再抽一个数据
            src, trg = self.db_manager.get_src_trg(
                random.randint(0, self.data_len))

        if self.enable_pretrain:
            # 自动构造数据训练
            # 可能出现正样本
          
            src = self.text_with_chr_confusion(trg)
       
        inputs = self.parse_line_data_and_label(src, trg)
        # build pos sample
        pos_inputs = self.parse_line_data_and_label(trg, trg)
        return_dict = {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_detect_tags': torch.LongTensor(inputs['token_detect_tags']),
            'sentence_detect_tags': torch.LongTensor(inputs['sentence_detect_tags']),
            'pos_input_ids': torch.LongTensor(pos_inputs['input_ids']),
            'pos_attention_mask': torch.LongTensor(pos_inputs['attention_mask']),
            'pos_token_detect_tags': torch.LongTensor(pos_inputs['token_detect_tags']),
            'pos_sentence_detect_tags': torch.LongTensor(pos_inputs['sentence_detect_tags']),
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

        # --- 对所有 token 计算loss ---
        src_len = len(src)
        ignore_loss_seq_len = self.max_seq_len-(src_len+1)  # sep and pad
        # 先默认给keep，会面对有错误标签的进行修改, ignore punc loss
        d_tags = [self._loss_ignore_id] + [self._keep_d_tag_id if not inclue_punc(
            c) else self._loss_ignore_id for c in src] + [self._loss_ignore_id] * ignore_loss_seq_len

        for (replace_idx, replace_char) in replace_idx_list:
            # +1 是因为input id的第一个字是cls
            d_tags[replace_idx+1] = self._error_d_tag_id
         
        
        if src == trg:
            sentence_detect_tags = self._keep_d_tag_id
        else:
            sentence_detect_tags = self._error_d_tag_id

        inputs['token_detect_tags'] = d_tags
        inputs['sentence_detect_tags'] = [sentence_detect_tags]
        
        return inputs

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

    def random_error_num_by_length_stragey(self, candidate_idx_len):
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
    d = DatasetCscLm(tokenizer,
                    max_seq_len=128,
                    ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                    db_dir='db/realise_train_ft',
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
