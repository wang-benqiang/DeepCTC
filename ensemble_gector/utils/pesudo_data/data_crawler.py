
import json
import os
import random
import time

import requests
from lxml import etree
from rich.progress import track
from utils.data_helper import include_cn, remove_order, remove_space


class SentenceSpider:
    def __init__(self) -> None:
        self.http_headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36',
                             'Content-Type': 'application/x-www-form-urlencoded'
                             }

        self.url = 'https://zaojv.com/wordQueryDo.php'
        self.form_data = {'wo': '截止', 'directGo': '1'}

    def request(self, trg_word, src_word, sentence_nums=3, with_words_confusion=False, error_log_fp=None):
        """[summary]

        Args:
            ori_word ([type]): [正确词语]
            error_word ([type]): [错误词语]
            sentence_nums (int, optional): [造句数量]. Defaults to 3.
            with_words_confusion (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        self.form_data['wo'] = trg_word
        rp = requests.post(url=self.url, data=self.form_data,
                           headers=self.http_headers, allow_redirects=True).text
        sentences = etree.HTML(rp).xpath('//*[@id="all"]//div')  # all
        if not sentences:
            sentences = etree.HTML(rp).xpath(
                '//*[@id="content"]//div')  # 造句网库里没有会从百度搜索相关句子。
        sentences = [remove_space(i.xpath('string(.)')) for i in sentences]
        sentences = [remove_order(i) for i in sentences if trg_word in i]
        src_texts, trg_texts = [], []

        if len(sentences) < 1:
            if error_log_fp is not None:
                error_log_fp.write(trg_word+'\n')
            print('[失败]   right_word:{} src_word:{},'.format(trg_word, src_word))

        print('[成功爬取到 {} 条]   right_word:{} src_word:{},'.format(
            len(sentences), trg_word, src_word))
        work_sent_num = 0
        for s in sentences:
            if trg_word in s:
                if not include_cn(s[-1]):

                    # 随机在末尾去掉符号，让模型不对句号敏感

                    trg = s
                    src = s.replace(trg_word, src_word)
                    random_seed = random.random()
                    if random_seed < 0.55:
                        trg = trg[:-1]
                        src = src[:-1]
                    trg_texts.append(trg)
                    src_texts.append(src)
                    work_sent_num += 1
                    if work_sent_num >= sentence_nums:
                        break
            elif error_log_fp is not None:
                error_log_fp.write(trg_word+'\n')
        # set && shuffle
        src_trg_li = list(zip(trg_texts, src_texts))
        src_trg_li = list(set(src_trg_li))
        random.shuffle(src_trg_li)
        # with_words_confusion
        word_sample = (trg_word, src_word)
        if with_words_confusion and word_sample not in src_trg_li:
            src_trg_li = [word_sample] + src_trg_li
        return src_trg_li[:sentence_nums]

    @staticmethod
    def load_json_file(fp):
        fp = open(fp, 'r', encoding='utf8')
        return json.load(fp)

    def crawler_data_from_confusion_set(self, error_log_fp, out_fp='data/artificial_data/crawl_csc/test.txt'):

        error_log_fp = open(error_log_fp, 'w', encoding='utf8')
        out_fp = open(out_fp, 'w', encoding='utf8')
        chengyu_confusion = self.load_json_file(
            'src/utils/pseudo_data/data/chengyu_confusion.json')
        words_confusion = self.load_json_file(
            'src/utils/pseudo_data/data/words_confusion.json')
        src_texts, trg_texts = [], []

        # 成语
        for trg_word, src_words in track(chengyu_confusion.items(), total=len(chengyu_confusion)):
            for src_word in src_words:
                # 把字词本身替换加上去
                src_texts.append(src_word)
                trg_texts.append(trg_word)

                res = self.request(trg_word, src_word,
                                   sentence_nums=5, error_log_fp=error_log_fp)
                if len(res) > 0:
                    srcs, trgs = list(zip(*res))
                    src_texts.extend(srcs), trg_texts.extend(trgs)
        # 词语
        for trg_word, src_words in track(words_confusion.items(), total=len(words_confusion)):
            for src_word in src_words:
                res = self.request(trg_word, src_word,
                                   sentence_nums=5, error_log_fp=error_log_fp)
                if len(res) > 0:
                    srcs, trgs = list(zip(*res))
                    src_texts.extend(srcs), trg_texts.extend(trgs)

        [out_fp.write('{}\t{}\n'.format(trg, src))
         for trg, src in zip(trg_texts, src_texts)]
        return 1

    def manual_crawl_data_to_file(self, right_word, wrong_word, data_fp=None, with_words_confusion=False, sample_num=5, task='csc'):

        if data_fp is None:
            # 如果没指定写入哪个文件，则按时间写入
            data_dir = 'data/artificial_data/{}/new'.format(task)
            current_time = time.strftime("%YY%mM%dD", time.localtime())
            data_fp = os.path.join(data_dir, '{}.txt'.format(current_time))

        print('data_fp:{}'.format(data_fp))

        if os.path.exists(data_fp):
            data_fp = open(data_fp, 'a', encoding='utf8')
        else:
            data_fp = open(data_fp, 'w', encoding='utf8')
        src_right_li = self.request(right_word, wrong_word,
                                    with_words_confusion=with_words_confusion, sentence_nums=sample_num)

        for trg, src in src_right_li:
            if task == 'csc':
                if len(src) == len(trg):
                    data_fp.write('{}\t{}\n'.format(trg, src))
                else:
                    print('trg(len:{}):{}'.format(len(trg), trg))
                    print('src(len:{}):{}'.format(len(src), src))
            elif task in ('gec', 'pos'):
                data_fp.write('{}\t{}\n'.format(trg, src))


if __name__ == '__main__':
    s = SentenceSpider()
    # r = s.crawler_data_from_confusion_set(error_log_fp='logs/miss_words.txt')
    s.manual_crawl_data_to_file(
        right_word='先辈',
        wrong_word='前辈',
        with_words_confusion=True,
        sample_num=5,
        task='csc')
    # print(r)
