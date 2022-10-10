from logging import raiseExceptions
import os
import pickle
from copy import deepcopy

import torch
from rich.progress import track
from src import logger
from src.utils.data_helper import replace_punc_for_bert
from torch.utils.data import Dataset
from transformers import BertTokenizer


class Seq2labelDataset(Dataset):
    def __init__(self,
                 data_dir,
                 tokenizer: BertTokenizer,
                 ctc_label_vocab_dir,
                 max_seq_len,
                 d_tag_type = 'all',
                 keep_one_append=False,
                 max_dataset_len=None,
                 ds_pkl_fp=None):
        """

        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(Seq2labelDataset, self).__init__()
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        d_tag_types = ('all', 'replace','redundant', 'miss')
        assert d_tag_type in d_tag_types, 'keep d_tag_type in {}'.format(d_tag_types)
        self.d_tag_type = d_tag_type
        self.max_dataset_len = max_dataset_len
        self.ds_pkl_fp = ds_pkl_fp
        self.tokenizer = tokenizer

        self.data_segments_list = self._read_ctc_data_from_file(
            keep_one_append)  # [ [['$START', '$KEEP'], ['明', '$KEEP']], ... ]

        self.id2dtag, self.dtag2id = self.load_label_dict(
            ctc_label_vocab_dir)
       
        self.dtag_num = len(self.dtag2id)
        if self.ds_pkl_fp is not None:
            self.save_ds_pkl()
        logger.info('dataset length: {}'.format(len(self.data_segments_list)))
        logger.info('dataset pkl saved at {}'.format(self.ds_pkl_fp))

    def __getitem__(self, item):
        inputs = self.parse_line_data_and_label(self.data_segments_list[item])
        return {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
            'd_tags': torch.LongTensor(inputs['d_tags']),
            'raw_data': inputs['raw_data'],
        }

    def __len__(self):
        return len(self.data_segments_list)

    def _read_ctc_data_from_file(self, keep_one_append, chunk_num=100000):
        """读取ctc数据

        Args:
            keep_one_append ([type]): 是否迭代生成多条数据
            chunk_num (int, optional): [description]. Defaults to 100000.

        Returns:
            list:[ [['$START', '$KEEP'], ['明', '$KEEP']], ...]
        """
        all_file = os.listdir(self.data_dir)
        all_file_fp = [
            '{}/{}'.format(self.data_dir, f_name) for f_name in all_file
        ]
        all_data_list = []
        print('loading {} files:{}'.format(len(all_file_fp), str(all_file_fp)))
        for file_fp in track(all_file_fp,
                             description='Processing...',
                             total=len(all_file_fp)):
            lines = [
                line.strip() for line in open(file_fp, 'r', encoding='utf8')
            ]
            
            if self.max_dataset_len is not None:
                if len(lines) > self.max_dataset_len:
                    lines = lines[0:self.max_dataset_len]
                

            for i in range(0, len(lines), chunk_num):
                list(
                    map(
                        lambda x: all_data_list.extend(
                            self.gen_data_from_line(x, keep_one_append)),
                        lines[i:i + chunk_num]))

            if self.max_dataset_len is not None:
                if len(all_data_list) > self.max_dataset_len:
                    all_data_list = all_data_list[0:self.max_dataset_len]
                    return all_data_list
        return all_data_list

    def load_label_dict(self, ctc_label_vocab_dir):
        if self.d_tag_type =='all':
            dtag_fp = '{}/{}'.format(ctc_label_vocab_dir, 'd_3tags.txt')
        elif self.d_tag_type =='redundant':
            dtag_fp = '{}/{}'.format(ctc_label_vocab_dir, 'redundant_tags.txt')
        elif self.d_tag_type =='miss':
            dtag_fp = '{}/{}'.format(ctc_label_vocab_dir, 'missing_tags.txt')
        elif self.d_tag_type =='replace':
            dtag_fp = '{}/{}'.format(ctc_label_vocab_dir, 'replace_tags.txt')
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id,

    def parse_line_data_and_label(self, data):
        # [[['$START', '$KEEP'], ['明', '$KEEP']], ...]
        # 保证传入的数据里没有空格如：[' ', '$KEEP']
        text = '始' + replace_punc_for_bert(''.join(
            list(map(lambda x: x[0], data))[1:]))  # 移除START

        text = ' '.join(text)  # 对英文字级别打token，防止单词之类的#号bug
        d_tags = list(map(lambda x: x[1].split('_')[0],
                            data))  # 对应词典，拿到dtag
        # 这里把对不上dtag变成了0 对应keep
        d_tags = [-1] + list(map(lambda i: self.dtag2id.get(i, 0), d_tags))
        # 这里把对不上Ctag变成了-1
        d_tags = d_tags[:self.max_seq_len]
        if len(d_tags) < self.max_seq_len:
            d_tags += [-1] * (self.max_seq_len - len(d_tags)
                                )  # pad的label ignore

        # 这里 start 先用 始 代替， 后面替换
        inputs = self.tokenizer(text,
                                padding=True,
                                truncation=True,
                                max_length=self.max_seq_len)
        inputs['input_ids'][1] = 2  # 这里将 始 替换为bert词典的[unused]
        
        diff_num = self.max_seq_len - len(inputs['input_ids'])
        inputs['input_ids'] = inputs['input_ids'] + [
            self.tokenizer.pad_token_id
        ] * diff_num

        inputs['attention_mask'] = inputs['attention_mask'] + [
            0
        ] * diff_num
        inputs['token_type_ids'] = inputs['token_type_ids'] + [
            0
        ] * diff_num
        
        inputs['d_tags'] = d_tags
        inputs['raw_data'] = ' '.join(self.tokenizer.convert_ids_to_tokens(inputs['input_ids']))
     

        return inputs

    def gen_data_from_line(self, line, keep_one_append=False):
        """
        从每一个line读取生成数据
        Args:
            line: line of train.txt
            keep_one_append: 只处理一个append or 多个
        Returns:
            [ [['$START', '$KEEP'], ['明', '$KEEP']], ... ]
        """
        line_segments = line.replace('SEPL__SEPR', ' SEPL__SEPR').split(' ')
        if keep_one_append:
            line_keep_one_append_segments = [
                i.split('SEPL|||SEPR') for i in line_segments
                if 'SEPL__SEPR$APPEND' not in i
            ]
            return [line_keep_one_append_segments]
        else:
            # 根据append数量迭代生成多条数据
            lines_segments = []
            default_line_segments = []  # 默认保留一个的append
            addition_append_line_segments = []  # 被丢弃的多个append
            for idx, segment in enumerate(line_segments):
                if 'SEPL__SEPR$APPEND' not in segment:
                    r = segment.split('SEPL|||SEPR')
                    default_line_segments.append(r)  # ['明', '$KEEP']
                else:
                    r = (idx, segment.split('SEPL__SEPR')
                         )  # (18, ['', '$APPEND_那'])
                    addition_append_line_segments.append(r)

            lines_segments.append(
                default_line_segments)  # 保存keep one append 的数据
            last_line_segments = deepcopy(
                default_line_segments)  # 记录每次append后的数据，迭代生成数据
            for addition_append_line_segments in addition_append_line_segments:
                new_line_segments = deepcopy(last_line_segments)
                idx, append_value = addition_append_line_segments[
                    0], addition_append_line_segments[1][1]  # 18, $APPEND_那
                if '$APPEND' in new_line_segments[idx - 1][1]:
                    new_line_segments.insert(idx, [
                        new_line_segments[idx - 1][1].split('$APPEND_')[-1],
                        append_value
                    ])
                    new_line_segments[idx - 1][1] = '$KEEP'
                    last_line_segments = new_line_segments
                else:
                    # 前面是replace
                    new_line_segments.insert(idx, [
                        new_line_segments[idx - 1][1].split('$REPLACE_')[-1],
                        append_value
                    ])
                    new_line_segments[idx - 1] = [
                        new_line_segments[idx - 1][1].split('$REPLACE_')[0],
                        '$KEEP'
                    ]
                    last_line_segments = new_line_segments

                lines_segments.append(new_line_segments)
            # r1 append
            #
            return lines_segments

    def write_data_to_file(self, fp='example_data/train/addition.txt'):

        with open(fp, 'w', encoding='utf8') as f:
            # 迭代出來的新數據添加原始文本
            [
                f.write(' '.join(map(lambda x: 'SEPL|||SEPR'.join(x), item)) +
                        '\n') for item in self.data_segments_list
            ]

    def save_ds_pkl(self):
        # 数据序列化保存
        f = open(self.ds_pkl_fp, 'wb')
        pickle.dump(self, f)
        logger.info('dataset pkl saved at {}'.format(self.ds_pkl_fp))
        f.close()

    @staticmethod
    def load_ds_pkl(ds_pkl_fp):
        f = open(ds_pkl_fp, 'rb')
        r = pickle.load(f)
        logger.info('dataset pkl loaded from {}'.format(ds_pkl_fp))
        f.close()
        return r


if __name__ == '__main__':
    data_path = './example_data/'
    tokenizer_path = './pretrained_model/electra_base_cn_discriminator'
    ctc_vocab_path = './src/recall/ctc_vocab'

    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    d = Seq2labelDataset(data_path, tokenizer, ctc_vocab_path, max_seq_len=156)
    dataset = torch.utils.data.dataloader.DataLoader(d, batch_size=10)
    # d.write_data_to_file()
    for i in dataset:
        print(i)
    print(d.data_segments_list)
