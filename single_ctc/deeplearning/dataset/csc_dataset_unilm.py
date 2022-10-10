import random
from difflib import SequenceMatcher

import torch
from src import logger
from src.deeplearning.dataset.lm_db.db_manager import CtcDBManager
from src.deeplearning.dataset.lm_db.yaoge_lmdb import TrainDataLmdb
from src.utils.data_helper import replace_punc_for_bert, inclue_punc
from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
from torch.utils.data import Dataset
import os


class DatasetCscUnilm(Dataset):
    "unilm for csc"

    def __init__(self,
                 tokenizer: CustomBertTokenizer,
                 max_seq_len,
                 ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                 max_dataset_len=None,
                 db_dir='src/deeplearning/dataset/lm_db/db/ms',
        
                 ):
        """
        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetCscUnilm, self).__init__()
        self.max_seq_len = max_seq_len
        
        self.max_dataset_len = max_dataset_len
        self.tokenizer = tokenizer
        self.id2dtag, self.dtag2id, self.id2ctag, self.ctag2id = self.load_label_dict(
            ctc_label_vocab_dir)

        # self.dtag_num = len(self.dtag2id)

        self.db_dir = db_dir

        if 'yaoge' in db_dir:
            self.db_manager = TrainDataLmdb(lmdb_dir=db_dir)
        else:
            self.db_manager = CtcDBManager(lmdb_dir=db_dir)
        self.data_len = len(self.db_manager)
        logger.info('db:{}, db len: {}'.format(
            self.db_dir, len(self.db_manager)))

        # 检测标签
        self._keep_d_tag_id, self._error_d_tag_id = self.dtag2id['$KEEP'], self.dtag2id['$ERROR']
        # self._keep_d_tag_id, self._replace_d_tag_id = self.dtag2id['$KEEP'], self.dtag2id['$REPLACE']
        # self._delete_d_tag_id, self._append_d_tag_id = self.dtag2id['$DELETE'], self.dtag2id['$APPEND']
        
        # voab id
       
    
        self._cls_vocab_id = self.tokenizer.vocab['[CLS]']
        self._sep_vocab_id = self.tokenizer.vocab['[SEP]']
        self._unk_vocab_id = self.tokenizer.vocab['[UNK]']
        # loss ignore id
        self._loss_ignore_id = -100

    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_2tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')
        
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        
        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id

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
                replace_idx_list += [ (i, '$REPLACE_'+trg_text[j]) for i, j in zip(range(i1, i2), range(j1, j2))]

            # elif tag == 'insert' and j2-j1 == 1:
            #     missing_idx_list.append((i1-1, '$APPEND_'+trg_text[j1]))
            # elif tag == 'delete':
            #     # 叠字叠词删除后面的
            #     redundant_length = i2-i1
            #     post_i1, post_i2 = i1+redundant_length, i2+redundant_length
            #     if src_text[i1:i2] == src_text[post_i1:post_i2]:
            #         i1, i2 = post_i1, post_i2
            #     for i in range(i1, i2):
            #         delete_idx_list.append(i)
            

        return keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list

    def __getitem__(self, item):
        src, trg = self.db_manager.get_src_trg(item)
        inputs = self.parse_line_data_and_label(src, trg)
        return_dict = {
            'raw_src':src,
            'raw_trg':trg,
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'c_tags': torch.LongTensor(inputs['c_tags']),
            'd_tags': torch.LongTensor(inputs['d_tags'])
        }
        return return_dict

    def __len__(self):
        if (self.max_dataset_len is not None and
                self.max_dataset_len < self.data_len):
            return self.max_dataset_len
        return self.data_len

    def convert_ids_to_ctags(self, ctag_id_list):
        return [self.id2ctag[i] if i!=self._loss_ignore_id else self._loss_ignore_id for i in ctag_id_list]
    
    def convert_ids_to_dtags(self, dtag_id_list):
        return [self.id2dtag[i] if i!=self._loss_ignore_id else self._loss_ignore_id for i in dtag_id_list]
    
    def parse_line_data_and_label(self, src, trg):
        """[summary]

        Args:
            src ([type]): text
            redundant_marks ([type]): [(1,2), (5,6)]

        Returns:
            [type]: [description]
        """
        if src and len(src) < 3:
            trg = src
        
        
        # 随机在句子末尾增加或删除标点，增加鲁棒性
        punc_action = random.choice([True, False])
        
        if punc_action and len(src)>3:
            if inclue_punc(src[-1]):
                src, trg = src[:-1], trg[:-1]
            else:
                # 随机选一个标点加到末尾
                random_punc = random.choice(['.','。','。','！', '!'])
                src, trg = src+random_punc, trg+random_punc
        src, trg = replace_punc_for_bert(src)[:self.max_seq_len - 2], replace_punc_for_bert(trg)[:self.max_seq_len - 2] # trg是用decoder预测,所以不用始开头, 只需要用eos结尾, 所以文本长度-1
    
        # 对英文字级别打token，防止单词之类的#号bug
        inputs = self.tokenizer(src,
                                max_len=self.max_seq_len)
        # 取单个value
        for k, v in inputs.items():
            inputs[k] = v[0]

        c_tags = [self.tokenizer.vocab.get(c, self._unk_vocab_id) for c in trg]
        c_tags_pad_len = self.max_seq_len-len(c_tags)-2
        c_tags = [self._loss_ignore_id] + c_tags + [self._loss_ignore_id] + [self._loss_ignore_id] * c_tags_pad_len
        
        # 计算detecot label
        keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list = self.match_ctc_idx(src, trg) # 对应前面处理src加了始字
      

        # --- 对所有 token 计算loss ---
        src_len = len(src)
        ignore_loss_seq_len= self.max_seq_len-(src_len+1)  # sep and pad
        # 先默认给keep，会面对有错误标签的进行修改
        d_tags = [self._loss_ignore_id] + [self._keep_d_tag_id] * src_len + [self._loss_ignore_id] * ignore_loss_seq_len
        
        for (replace_idx, replace_char) in replace_idx_list:
            # +1 是因为input id的第一个字是cls
            d_tags[replace_idx+1] = self._error_d_tag_id
        
        # for delete_idx in delete_idx_list:
        #     d_tags[delete_idx+1] = self._error_d_tag_id

        # for (miss_idx, miss_char) in missing_idx_list:
        #     d_tags[miss_idx + 1] = self._error_d_tag_id

        
        inputs['d_tags'] = d_tags
        inputs['c_tags'] = c_tags
        return inputs



if __name__ == '__main__':
    
    tokenizer_path = 'pretrained_model/bart-base-chinese-cluecorpussmall'
    tokenizer = CustomBertTokenizer.from_pretrained(tokenizer_path)
    d = DatasetCscUnilm(tokenizer,
                       max_seq_len=128,
                       ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                       db_dir='data/train_data_csc_yaoge_0223/train_csc_lmdb_0207_1')
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=128)
    for i in dataset:
        print(i)

    src_text = '可老爸还是无动于束'
    trg_text = '可老爸还是无动于衷'
    
    r = d.match_ctc_idx(src_text, trg_text)
    
    r = d.parse_line_data_and_label(src_text, trg_text)
    x = iter(d)
    r = next(x)
    print(r)
