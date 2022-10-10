import os
from sklearn.utils import shuffle

import torch
from logs import logger
from src.tokenizer.bert_tokenizer import CustomBertTokenizer
from torch.utils.data import Dataset
from utils.data_helper import replace_punc_for_bert, include_cn, inclue_punc
from utils.lmdb.db_manager import CtcDBManager
from utils.lmdb.yaoge_lmdb import TrainDataLmdb
import random
from difflib import SequenceMatcher
from utils.gector_preprocess_data import align_sequences
from transformers.data.data_collator import DataCollatorForWholeWordMask


class DatasetCtcSeq2Edit(Dataset):
    def __init__(self,
                 tokenizer: CustomBertTokenizer,
                 max_seq_len,
                 max_dataset_len=-1,
                 db_dir='data/lm_db/db_test',
                 ctc_label_vocab_dir='src/vocab'
                 ):
        """

        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetCtcSeq2Edit, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len
        self.tokenizer = tokenizer
        self.id2dtag, self.dtag2id, self.id2ctag, self.ctag2id = self.load_label_dict(
            ctc_label_vocab_dir)
        self.db_dir = db_dir
        if self.db_dir is not None and os.path.exists(self.db_dir):
            if 'yaoge' in self.db_dir:
                self.db_manager = TrainDataLmdb(lmdb_dir=self.db_dir)
            else:
                self.db_manager = CtcDBManager(lmdb_dir=self.db_dir)
        

        self.data_len = len(self.db_manager)
        logger.info('all samples loaded, num: {}'.format(
            self.data_len))

        self.punc_list_in_end = ['.', '。', '。', '！', '!']
        self.punc_list_in_end = self.punc_list_in_end + \
            [i*2 for i in self.punc_list_in_end]  # 标点数量乘2
        
        
        # vocab id
        self._keep_d_tag_id, self._error_d_tag_id = self.dtag2id['$RIGHT'], self.dtag2id['$ERROR']
        self._keep_c_tag_id = self.ctag2id['$KEEP']
        self._delete_c_tag_id = self.ctag2id['$DELETE']
        self.replace_unk_c_tag_id = self.ctag2id['[REPLACE_UNK]']
        self.append_unk_c_tag_id = self.ctag2id['[APPEND_UNK]']
        self.unk_c_tag_id = self.ctag2id['[UNK]']
        self._loss_ignore_id = -100
    
    
    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')
        
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        
        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id
    
    def convert_ids_to_ctags(self, ctag_id_list):
        return [self.id2ctag[i] if i!=self._loss_ignore_id else self._loss_ignore_id for i in ctag_id_list]
    
    def convert_ids_to_dtags(self, dtag_id_list):
        return [self.id2dtag[i] if i!=self._loss_ignore_id else self._loss_ignore_id for i in dtag_id_list]
    
    @staticmethod
    def match_ctc_idx(src_text, trg_text):
        keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list = [], [], [], []
        
        r = SequenceMatcher(None, src_text, trg_text)
        diffs = r.get_opcodes()

        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            
      
            if tag == 'replace' and i2-i1 == j2-j1:
                # 如果文本中出现连续不同的错误label，会被diff库直接处理成replace操作
                # 所以暂时确保只有是错字类型再输出label
                replace_idx_list += [ (i, '$REPLACE_'+trg_text[j]) for i, j in zip(range(i1, i2), range(j1, j2))]

            elif tag == 'insert' and j2-j1 == 1:
                missing_idx_list.append((i1-1, '$APPEND_'+trg_text[j1]))
            elif tag == 'delete':
                # 叠字叠词删除后面的
                redundant_length = i2-i1
                post_i1, post_i2 = i1+redundant_length, i2+redundant_length
                if src_text[i1:i2] == src_text[post_i1:post_i2]:
                    i1, i2 = post_i1, post_i2
                for i in range(i1, i2):
                    delete_idx_list.append(i)
            

        return keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list
    
    
    def parse_label_id_list_by_gectorway(self, src_text, trg_text, max_len):
        
        labels = align_sequences(src_text, trg_text)
        d_labels = [] 
        c_labels = []
        true_length = len(labels)
        for label_list in labels:
            c_labels.append(self.ctag2id.get(label_list[0], self.unk_c_tag_id))
            d_labels.append(0 if 'KEEP' in label_list[0] else 1)
        d_labels += [self._loss_ignore_id] * (max_len-true_length)
        c_labels += [self._loss_ignore_id] * (max_len-true_length)
        
        return d_labels, c_labels
    
    def gen_src_trg_iteration(self, src_text, trg_text):
        
        labels = align_sequences(src_text, trg_text)

        todo_iteration_dict = []
        src_text_li = [''] + list(src_text)
        
        new_src_text_li = []
        for idx, label_list in enumerate(labels):
            if len(label_list) >1 :
                todo_iteration_dict.append([idx, label_list[1:]])
            first_label = label_list[0]
            
            if 'KEEP' in first_label:
                new_src_text_li.append(src_text_li[idx])
            elif 'REPLACE_' in first_label:
                token=first_label.split('REPLACE_')[-1]
                new_src_text_li.append(token)
            elif 'APPEND_' in first_label:
                token=first_label.split('APPEND_')[-1]
                new_src_text_li.append(src_text_li[idx]+token)
        
        if len(todo_iteration_dict)>0:
            new_src_text_list = []
            new_trg_text_list = []
            for idx, label_list in todo_iteration_dict:
                for label in label_list:
                    this_time_src_text = ''.join(new_src_text_li)
                    new_src_text_list.append(this_time_src_text)
                    assert 'APPEND_' in label
                    token=label.split('APPEND_')[-1]
                    new_src_text_li[idx] = new_src_text_li[idx]+token
                    this_time_trg_text = ''.join(new_src_text_li)
                    new_trg_text_list.append(this_time_trg_text)
    
        
        return new_src_text_list, new_trg_text_list
    
    def match_disorder_label(self, src_text, trg_text):
        """返回需要删除的索引范围, 叠字叠词删除靠后的索引

        Args:
            src ([type]): [description]
            trg ([type]): [description]

        Returns:
            [type]: [(1,2), (5,7)]
        """

        # new_src_char_list = deepcopy(src_char_list)
        if len(src_text) != len(trg_text):
            return []
            
        keep_id, left_id, right_id = self.ctag2id['$KEEP'], self.ctag2id['$LEFT'], self.ctag2id['$RIGHT']
        label = [keep_id] * len(src_text)
        res = SequenceMatcher('', src_text, trg_text).get_matching_blocks()
        for sub_res in res:
            
            a, b, length = sub_res.a, sub_res.b, sub_res.size
            if a < b:
                # src: 222211  长串在src前面面
                # trg: 112222
                pred_text = src_text[:a] + trg_text[a:b] + \
                    trg_text[b:b+length] + src_text[b+length:]

                if pred_text == trg_text:  # 如果交换完=trg
                    label[a:a+length] = [left_id] * length
                    label[a+length:a+length+(b-a)] = [right_id] * (b-a)
                # 发现一个就break, 一句话只有一个乱序
                break

            elif a > b:
                
                # src: 112222  长串在src后面
                # trg: 222211
                pred_text = src_text[:b] + src_text[a:a +
                                                    length] + src_text[b:a] + src_text[a+length:]
                if pred_text == trg_text:  # 如果交换完=trg
                    label[b:a] = [left_id] * (a-b)
                    label[a:a+length] = [right_id] * length
                # 发现一个就break, 一句话只有一个乱序
                break
        return label
    
    def __getitem__(self, item):
        src_text, trg_text = self.db_manager.get_src_trg(item)
        
        while not include_cn(src_text):
            # 如果数据中不包含文本.随机再抽一个数据
            src_text, trg_text = self.db_manager.get_src_trg(random.randint(0, self.data_len))
        # if abs(len(src_text) - len(trg_text)) >3:
        #     src_text = trg_text
        
        # random select between pos text pair and neg text pair 
       
        
        # inputs = self.parse_line_data_and_label(src_text, trg_text)
        inputs = self.parse_data_by_gector(src_text, trg_text)

        return_dict = {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
            'd_labels': torch.LongTensor(inputs['d_tags']),
            'c_labels': torch.LongTensor(inputs['c_tags'])
        }

        return return_dict

    def __len__(self):
        if self.max_dataset_len != -1 and self.max_dataset_len < self.data_len:
            return self.max_dataset_len
        return self.data_len

    def parse_line_data_and_label(self, src_text, trg_text):

        
        # 随机在句子末尾增加或删除标点，增加鲁棒性
        punc_action = random.choice([True, False])

        if punc_action and len(src_text) > 3:
            if inclue_punc(src_text[-1]):
                src_text, trg_text = src_text[:-1], trg_text[:-1]
            else:
                # 随机选一个标点加到末尾
                random_punc = random.choice(self.punc_list_in_end)
                src_text, trg_text = src_text+random_punc, trg_text+random_punc
        
        
        src_text, trg_text = replace_punc_for_bert(
            src_text)[:self.max_seq_len - 2], replace_punc_for_bert(trg_text)[:self.max_seq_len - 2]
        # 对英文字级别打token，防止单词之类的#号bug

        inputs = self.tokenizer(src_text, max_len=self.max_seq_len)
        for k, v in inputs.items():
            inputs[k] = v[0]
        
        
        
        # --- 对所有 token 计算loss ---
        src_len = len(src_text)
        ignore_loss_seq_len= self.max_seq_len-(src_len+1)  # sep and pad
        # 先默认给keep，会面对有错误标签的进行修改, 没有忽略标点的loss
        d_tags = [self._loss_ignore_id] + [self._keep_d_tag_id] * src_len + [self._loss_ignore_id] * ignore_loss_seq_len
        c_tags = [self._loss_ignore_id] + [self._keep_c_tag_id] * src_len + [self._loss_ignore_id] * ignore_loss_seq_len
        
        # 先匹配乱序
        
        disorder_label_list = self.match_disorder_label(src_text, trg_text)
        if 3 in disorder_label_list and 4 in disorder_label_list:
            # 有乱序的情况
            for idx, c_label in enumerate(disorder_label_list):
                d_tags[idx+1] = self._error_d_tag_id
                c_tags[idx+1] = c_label
        else:
            
            keep_idx_list, replace_idx_list, delete_idx_list, missing_idx_list = self.match_ctc_idx(src_text, trg_text)
            
            for (replace_idx, replace_char) in replace_idx_list:
                # +1 是因为input id的第一个字是cls
                d_tags[replace_idx+1] = self._error_d_tag_id
                c_tags[replace_idx+1] = self.ctag2id.get(replace_char, self.replace_unk_c_tag_id)
            
            for delete_idx in delete_idx_list:
                d_tags[delete_idx+1] = self._error_d_tag_id
                c_tags[delete_idx+1] = self._delete_c_tag_id

            for (miss_idx, miss_char) in missing_idx_list:
                d_tags[miss_idx + 1] = self._error_d_tag_id
                c_tags[miss_idx +
                    1] = self.ctag2id.get(miss_char, self.append_unk_c_tag_id)
        inputs['d_tags'] = d_tags
        inputs['c_tags'] = c_tags
        return inputs
    
    def parse_data_by_gector(self, src_text, trg_text):

        
        # 随机在句子末尾增加或删除标点，增加鲁棒性
        punc_action = random.choice([True, False])

        if punc_action and len(src_text) > 3:
            if inclue_punc(src_text[-1]):
                src_text, trg_text = src_text[:-1], trg_text[:-1]
            else:
                # 随机选一个标点加到末尾
                random_punc = random.choice(self.punc_list_in_end)
                src_text, trg_text = src_text+random_punc, trg_text+random_punc
        
        
        src_text, trg_text = replace_punc_for_bert(
            src_text)[:self.max_seq_len - 2], replace_punc_for_bert(trg_text)[:self.max_seq_len - 2]
        # 对英文字级别打token，防止单词之类的#号bug
        d_tags, c_tags = self.parse_label_id_list_by_gectorway(src_text, trg_text, self.max_seq_len)
        
        inputs = self.tokenizer(src_text, max_len=self.max_seq_len)
        for k, v in inputs.items():
            inputs[k] = v[0]
        inputs['input_ids'][0] = 1  # cls to unused
        
        inputs['d_tags'] = d_tags
        inputs['c_tags'] = c_tags
        return inputs


if __name__ == '__main__':
    tokenizer_path = 'pretrained_model/gpt2_cn_cluesmall'

    tokenizer = CustomBertTokenizer.from_pretrained(tokenizer_path)
    d = DatasetCtcSeq2Edit(
        tokenizer=tokenizer,
        max_seq_len=128,
        max_dataset_len=1000,
        db_dir='data/lmdb/track2_cged_train',
    )
    # s = '望贵方快速回复'
    # # t = '望贵方尽快答复'
    # s = '众所都智，重视绿色食品越来越强调。'
    # t = '众所周知，现在越来越重视绿色食品。'
    dataset_collate = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    # r = d.parse_data_by_gector(s, t)
    # r = d.gen_src_trg_iteration(s, t)
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=128, shuffle=True, collate_fn=dataset_collate.torch_call)
    for i in dataset:
        print(i)

