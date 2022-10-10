import json
import random

import torch
from src import logger
from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
from src.utils.data_helper import inclue_punc, replace_punc_for_bert
from torch.utils.data import Dataset


class DatasetTc(Dataset):
    

    def __init__(self,
                 tokenizer: CustomBertTokenizer,
                 max_seq_len,
                 tc_label_vocab_fp='src/deeplearning/ctc_vocab/tc_tags.txt',
                 max_dataset_len=None,
                 data_fp='data/text_classification/news_and_law_0.json',
                 ):
        """
        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetTc, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len
        self.tokenizer = tokenizer
        self.id2tag, self.tag2id = self.load_label_dict(tc_label_vocab_fp)
        self.tag_num = len(self.tag2id)
        # voab id

        self._cls_vocab_id = self.tokenizer.vocab['[CLS]']
        self._sep_vocab_id = self.tokenizer.vocab['[SEP]']
        self.data_list = self.load_data(data_fp)
        self.data_len = len(self.data_list)
        self.punc_list_in_end = ['.', '。', '。', '！', '!']
        self.punc_list_in_end = self.punc_list_in_end + \
            [i*2 for i in self.punc_list_in_end]  # 标点数量乘2

        self._loss_ignore_id = -100

    def load_label_dict(self, tc_label_vocab_fp):

        id2tag = [line.strip() for line in open(
            tc_label_vocab_fp, encoding='utf8')]
        tag2id = {v: i for i, v in enumerate(id2tag)}
        logger.info('tc_tag num: {}, tc_tags:{}'.format(len(id2tag), id2tag))

        return id2tag, tag2id

    @staticmethod
    def load_data(data_fp):
        data_json = json.load(open(data_fp, 'r', encoding='utf8'))
        return data_json['data']

    def __getitem__(self, item):
        ele = self.data_list[item]
        content, labels = ele['content'], ele['label']

        inputs = self.parse_line_data_and_label(content, labels)
        return_dict = {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
            'labels': torch.FloatTensor(inputs['labels']),
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

    def parse_line_data_and_label(self, content, labels):
        """[summary]

        Args:
            src ([type]): text
            redundant_marks ([type]): [(1,2), (5,6)]

        Returns:
            [type]: [description]
        """

        # 随机在句子末尾增加或删除标点，增加鲁棒性
        punc_action = random.choice([True, False])

        if punc_action and len(content) > 3:
            if inclue_punc(content[-1]):
                content = content[:-1]
            else:
                # 随机选一个标点加到末尾
                random_punc = random.choice(self.punc_list_in_end)
                content = content+random_punc

        content = replace_punc_for_bert(
            content)[:self.max_seq_len - 2]
        # 对英文字级别打token，防止单词之类的#号bug
        inputs = self.tokenizer(content,
                                max_len=self.max_seq_len)
        # 取单个value
        for k, v in inputs.items():
            inputs[k] = v[0]

        labels_id = [0] * self.tag_num

        for label in labels:
            labels_id[self.tag2id.get(label, self.tag2id['[UNK]'])] = 1

        inputs['labels'] = labels_id
        return inputs


if __name__ == '__main__':

    tokenizer_path = 'pretrained_model/chinese-bert-wwm'
    tokenizer = CustomBertTokenizer.from_pretrained(tokenizer_path)
    d = DatasetTc(tokenizer,
                  max_seq_len=128,
                  )
    dataset = torch.utils.data.dataloader.DataLoader(
        d, batch_size=158, shuffle=True)
    for i in dataset:
        print(i)

    src_text = '可老爸还是无动于束'
    trg_text = '可老爸还是无动于衷'

    r = d.match_ctc_idx(src_text, trg_text)

    r = d.parse_line_data_and_label(src_text, trg_text)
    x = iter(d)
    r = next(x)
    print(r)
