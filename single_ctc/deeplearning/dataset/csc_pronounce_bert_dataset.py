import json
import os
import random
from difflib import SequenceMatcher

import torch
from src import logger
from src.deeplearning.dataset.lm_db.db_manager import CtcDBManager
from src.deeplearning.dataset.lm_db.yaoge_lmdb import TrainDataLmdb
from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
from src.utils.data_helper import (include_cn, inclue_punc,
                                   replace_punc_for_bert)
from torch.utils.data import Dataset
import multiprocessing as mp



class GlobalVars:
    char_pronounce_confusion = json.load(open('src/data/pinyin_data/char_pronounce_confusion.json', 'r', encoding='utf8'))
    char_confustion_iters=mp.Manager().dict({ char:iter(confusion) for char,confusion in char_pronounce_confusion.items()}) 
    char_replace_time_distribution= mp.Manager().dict({ char:0 for char in char_pronounce_confusion.keys()}) 
    char_replace_average_time=mp.Value('d',0)
    

class Value:
    # follow format of multipleprocess, char_replace_average_time
    def __init__(self, v):
        self.value = v

class DatasetCscSeq2labelSimPinyin(Dataset):


    def __init__(self,
                 tokenizer: CustomBertTokenizer,
                 max_seq_len,
                 ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                 max_dataset_len=None,
                 db_dir='src/deeplearning/dataset/lm_db/db/ms',
                 enable_pretrain=True,
                 char_confustion_iters=None,
                 char_replace_time_distribution=None,
                 char_replace_average_time=None,
                 ):
        
        """
        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含detect_labels.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetCscSeq2labelSimPinyin, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len
        self.tokenizer = tokenizer
        self.id2dtag, self.dtag2id, self.id2ctag, self.ctag2id, self.char2shengmu_dict, self.char2yunmu_dict, self.char_pronounce_confusion = self.load_label_dict(
            ctc_label_vocab_dir)
        self.char_pronounce_confusion_dict = {k:[i[0] for i in v] for k, v in self.char_pronounce_confusion.items()}
        if char_confustion_iters is not None:
            # load previous confusion distribution
            self.char_confustion_iters = char_confustion_iters
            self.char_replace_time_distribution = char_replace_time_distribution
            self.char_replace_average_time = char_replace_average_time
            logger.info('[confusion distribution] load confusion distribution from params')
        else:
            # reinit
            self.char_confustion_iters= { char:iter(confusion) for char,confusion in self.char_pronounce_confusion.items()}
            
            self.char_replace_time_distribution =  { char:0 for char in self.char_pronounce_confusion.keys()}
            
            self.char_replace_average_time = Value(0)
            logger.info('[confusion distribution] init')
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
        self._cn_vocab = [char for (char, idx) in tokenizer.vocab.items() if len(char)==1 and include_cn(char)]
        
        self._loss_ignore_id = -100

    
    def update_char_replace_average_time(self,):
        self.char_replace_average_time.value = sum([ times for char, times in self.char_replace_time_distribution.items()]) / len(self.char_replace_time_distribution)

    
    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_2tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')
        
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        
        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        
        
        # char to pinyin code
        char2shengmu_dict = json.load(open('src/data/pinyin_data/char_shengmu.json', 'r', encoding='utf8'))
        char2yunmu_dict = json.load(open('src/data/pinyin_data/char_yunmu.json', 'r', encoding='utf8'))
        char_pronounce_confusion = json.load(open('src/data/pinyin_data/char_pronounce_confusion.json', 'r', encoding='utf8'))
        
        
        char2shengmu_dict['[UNK]']
        # pinyin code to id 
        pycode2id = {
            str(v):idx for idx,v in enumerate(list(range(0, 10)) + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        }
        length = len(pycode2id)
        pycode2id['[PAD]'] = length
        pycode2id['[UNK]'] = length+1
        
        char2shengmu_dict = {char:pycode2id.get(code, pycode2id['[UNK]']) for char, code in  char2shengmu_dict.items()}
        char2yunmu_dict = {char:pycode2id.get(code, pycode2id['[UNK]']) for char, code in  char2yunmu_dict.items()}
      
        logger.info('d_tag num: {}, detect_labels:{}'.format(len(id2dtag), d_tag2id))
        
        return id2dtag, d_tag2id, id2ctag, c_tag2id, char2shengmu_dict, char2yunmu_dict, char_pronounce_confusion

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


        return keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list

    def __getitem__(self, item):
        src, trg = self.db_manager.get_src_trg(item)
        while not include_cn(src):
            # 如果数据中不包含文本.随机再抽一个数据
            src, trg = self.db_manager.get_src_trg(random.randint(0, self.data_len))
            
        # finetune also care autobuild mode
        auto_build_prob = random.random()
        if self.enable_pretrain or len(src) != len(trg) or auto_build_prob <= 0.3:
            # build nagative sample when len(src)!=len(trg)
            src  = self.text_with_chr_confusion(trg)
   
        elif src!=trg:
            # finetune mode, supervised learning
            src_char_li = list(trg)
            for idx, (src_char, trg_char) in enumerate(zip(src, trg)):
                if src_char!=trg_char and trg_char in self.char_pronounce_confusion_dict:
                    if src_char in self.char_pronounce_confusion_dict[trg_char]:
                        src_char_li[idx] = src_char
                    else:
                        try:
                            src_char_li[idx] = next(self.char_confustion_iters[trg_char])[0]
                        except Exception as e:
                            self.char_confustion_iters[trg_char] = iter(self.char_pronounce_confusion[trg_char])
                            src_char_li[idx] = next(self.char_confustion_iters[trg_char])[0]
            
            src = ''.join(src_char_li)
            
                
            
        inputs = self.parse_line_data_and_label(src, trg)
        
        return_dict = {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
            'detect_labels': torch.LongTensor(inputs['detect_labels']),
            'correct_labels': torch.LongTensor(inputs['correct_labels']),
            'input_shengmu': torch.LongTensor(inputs['input_shengmu']),
            'input_yunmu': torch.LongTensor(inputs['input_yunmu']),
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
        keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list = self.match_ctc_idx(src, trg)


        # --- 对所有 token 计算loss ---
        src_len = len(src)
        ignore_loss_seq_len= self.max_seq_len-(src_len+1)  # sep and pad
        # 先默认给keep，会面对有错误标签的进行修改
        detect_labels = [self._loss_ignore_id] + [self._keep_d_tag_id if not inclue_punc(c) else self._loss_ignore_id for c in src] + [self._loss_ignore_id] * ignore_loss_seq_len
        correct_labels = [self._loss_ignore_id] + [self._keep_c_tag_id if not inclue_punc(c) else self._loss_ignore_id for c in src] + [self._loss_ignore_id] * ignore_loss_seq_len
 
        input_shengmu = [self.char2shengmu_dict['[PAD]']] + [self.char2shengmu_dict.get(char, self.char2shengmu_dict['[UNK]']) if  include_cn(char) else self.char2shengmu_dict['[UNK]'] for char in src] + [self.char2shengmu_dict['[PAD]']] * ignore_loss_seq_len

        input_yunmu = [self.char2yunmu_dict['[PAD]']] + [self.char2yunmu_dict.get(char, self.char2yunmu_dict['[UNK]']) if  include_cn(char) else self.char2yunmu_dict['[UNK]'] for char in src] + [self.char2yunmu_dict['[PAD]']] * ignore_loss_seq_len
        
        for (replace_idx, replace_char) in replace_idx_list:
            # +1 是因为input id的第一个字是cls
            detect_labels[replace_idx+1] = self._error_d_tag_id
            correct_labels[replace_idx+1] = self.ctag2id.get(replace_char, self.replace_unk_c_tag_id)
        
        for delete_idx in delete_idx_list:
            detect_labels[delete_idx+1] = self._error_d_tag_id
            correct_labels[delete_idx+1] = self._delete_c_tag_id

        for (miss_idx, miss_char) in missing_idx_list:
            detect_labels[miss_idx + 1] = self._error_d_tag_id
            correct_labels[miss_idx +
                   1] = self.ctag2id.get(miss_char, self.append_unk_c_tag_id)

        
        inputs['detect_labels'] = detect_labels
        inputs['correct_labels'] = correct_labels
        inputs['input_shengmu'] = input_shengmu
        inputs['input_yunmu'] = input_yunmu
        return inputs

   

    
    def text_with_chr_confusion(self, correct_text):
        
        
        error_text = list(correct_text)
        
        candidate_idx = [idx for idx, char in enumerate(correct_text) if include_cn(char) and char in self.char_replace_time_distribution]
        
        candidate_idx_len = len(candidate_idx)
        
        # if candidate_idx_len == 0 :
        #     # if no candidate char, reduce constraints
        #     candidate_idx = [idx for idx, char in enumerate(correct_text) if include_cn(char) and char in self.char_replace_time_distribution]
            

        
        if candidate_idx_len > 0:
        
            error_num = max(0, self.random_error_num_by_length_stragey(candidate_idx_len))
            
            select_idx_li = random.sample(candidate_idx, error_num)

            random_select = random.choice([False,False,False,False,False,False,False,False,False,True])
   
            for select_idx in select_idx_li:
                select_char  = error_text[select_idx]
                if not random_select:
                    # average distribution
                    try:
                        error_text[select_idx] = next(self.char_confustion_iters[select_char])[0]
                    except:
                        # reload iteration
                        self.char_confustion_iters[select_char] = iter(self.char_pronounce_confusion[select_char])
                        error_text[select_idx] = next(self.char_confustion_iters[select_char])[0]
                else:
                    error_text[select_idx] = random.choice(self.char_pronounce_confusion[select_char])[0]
                        
                # update char relace time
                self.char_replace_time_distribution[select_char] += 1
                
                # self.update_char_replace_average_time()
                # print(self.char_replace_average_time)
        return ''.join(error_text)

    
    def random_error_num_by_length_stragey(self, candidate_idx_len):
        stragey = {
            
            range(1, 2): [0, 1],
            range(2, 5): [0, 1, 1, 1, 1, 1],
            range(5, 12): [1, 1, 1, 1, 2, 2],
            range(12, 32): [1, 1, 1, 2, 2, 2, 3],
            range(32, 64): [1, 1, 1, 2, 2, 2, 3, 3, 4],
            range(64, 129): [1, 1, 1,  2,2,2, 2, 3, 3, 3, 4, 5],
        }
        for len_range, error_num_li in stragey.items():
            if candidate_idx_len in len_range:
                return random.choice(error_num_li)
        return 0
    
if __name__ == '__main__':

    tokenizer_path = 'model/extend_electra_base'
    tokenizer = CustomBertTokenizer.from_pretrained(tokenizer_path)
    d = DatasetCscSeq2labelSimPinyin(tokenizer,
                                    max_seq_len=128,
                                    ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                                    db_dir='db/realise_train_ft',
                                    enable_pretrain=False)
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=158)
    for i in dataset:
        print(i)

    src_text = '可老爸还是无痘于束'
    trg_text = '可老爸还是无动于衷'
    
    r = d.match_ctc_idx(src_text, trg_text)
    
    r = d.parse_line_data_and_label(src_text, trg_text)
    x = iter(d)
    r = next(x)
    print(r)
