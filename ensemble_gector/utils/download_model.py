import argparse
import ftplib
import os
import sys
import zipfile

from tqdm import tqdm

# from data_downloader.downloader import Downloader




class Downloader(object):
    """自动从ftp server拉取数据或模型"""

    def __init__(self, local_path,
                 remote_path,
                 host='10.100.2.75',
                 username='yjy.public',
                 password='midu123.com',):
        """
        Args:
            local_path: str 本地路径
            remote_path: str 远程路径
            host:
        """
        self._local_path = local_path
        self._remote_path = remote_path
        self._host = host
        self._username = username
        self._password = password

    def _download(self):
        sys.stdout.write(
            '\nstart to download {} from ftp ...\n'.format(self._local_path))
        ftp = ftplib.FTP(self._host, user=self._username,
                         passwd=self._password, timeout=60)
        file_size = ftp.size(self._remote_path)
        file = open(self._local_path, "wb")
        process_bar = tqdm(unit='blocks', unit_scale=True, leave=False, miniters=1, desc='Downloading......',
                           total=file_size, mininterval=2)

        def bar(data):
            file.write(data)
            process_bar.update(len(data))

        ftp.retrbinary("RETR {}".strip().format(self._remote_path), bar)
        sys.stdout.write('success, save {} to {}\n'.format(
            self._remote_path, self._local_path))

        file.close()
        ftp.quit()

    def _unzip(self):
        zip_file = zipfile.ZipFile(self._local_path, 'r')
        zip_file.extractall(os.path.dirname(self._local_path))
        zip_file.close()
        os.remove(self._local_path)

    def get_path(self):
        local_path = self._local_path[:-4] if self._local_path.endswith(
            '.zip') else self._local_path
        if not os.path.exists(local_path):

            retry_time = 0
            while retry_time <= 3:
                try:
                    self._download()
                    break
                except Exception as e:
                    if os.path.exists(self._local_path):
                        os.remove(self._local_path)
                    retry_time += 1
                    if retry_time > 3:
                        raise Exception('download failed!')
                    sys.stdout.write(
                        '\n{}, start to retry: {}\n'.format(e, retry_time))

            if self._local_path.endswith('.zip'):
                self._unzip()
            sys.stdout.write('\n')
        return local_path


pretrained_model_dl = {
    "lang8":
    Downloader("data/ccl_2022/track_1_train/lang8.train.ccl22.zip",
               "/public/nlp/yjy-gen-corrector/lang8.train.ccl22.zip"),
    "sighan":
    Downloader("data/ccl_2022/track_1_train/sighan+wang.ccl22.zip",
               "/public/nlp/yjy-gen-corrector/sighan+wang.ccl22.zip"),
    
    "csc_data_v3":
    Downloader("data/train_data_csc_10e/data_1e5.zip",
               "/public/nlp/Chinese_Corpus/csc_train_data/data_v3/train_lmdb_1e5.zip"),
    "roberta":
    Downloader("pretrained_model/RoBERTa_zh_L12_PyTorch.zip",
               "/public/nlp/pretrained_model/RoBERTa_zh_L12_PyTorch.zip"),
    "roberta-large":
    Downloader("pretrained_model/roberta-large.zip",
               "/public/nlp/pretrained_model/roberta-large.zip"),
    "bart":
    Downloader("pretrained_model/bart_base_chinese_cluecorpussmall.zip",
               '/public/nlp/pretrained_model/bart-base-chinese-cluecorpussmall.zip'),
    "macbert":
    Downloader("pretrained_model/chinese-macbert-base.zip",
               "/public/nlp/pretrained_model/chinese-macbert-base.zip"),
    "macbert4csc":
    Downloader("pretrained_model/macbert4csc-base-chinese.zip",
               "/public/nlp/pretrained_model/macbert4csc-base-chinese.zip"),
    "mengzi-t5-base":
    Downloader("pretrained_model/mengzi-t5-base.zip",
               "/public/nlp/pretrained_model/mengzi-t5-base.zip"),
    "extend_macbert_base":
        Downloader("pretrained_model/extend_macbert_base.zip",
                   "/public/nlp/pretrained_model/extend_macbert_base.zip"),
    "wwm":
    Downloader("pretrained_model/chinese-bert-wwm.zip",
               "/public/nlp/pretrained_model/chinese-bert-wwm.zip"),
    "electra":
    Downloader(
        'pretrained_model/electra-base-180g-discriminator.zip',
        '/public/nlp/pretrained_model/electra-base-180g-discriminator.zip'),
    # electra_small.zip
    "electra_small":
    Downloader(
        'pretrained_model/electra_small.zip',
        '/public/nlp/pretrained_model/electra_small.zip'),
    "electra-large":
    Downloader(
        'pretrained_model/electra-180g-large-discriminator.zip',
        '/public/nlp/pretrained_model/electra-180g-large-discriminator.zip'),
    "macbert-large":
    Downloader("pretrained_model/chinese-macbert-large.zip",
               "/public/nlp/pretrained_model/chinese-macbert-large.zip"),
    "robert-wwm-ext":
    Downloader("pretrained_model/chinese-roberta-wwm-ext.zip",
               "/public/nlp/pretrained_model/chinese-roberta-wwm-ext.zip"),
    "realise":
    Downloader("pretrained_model/realise_well_trained_model.zip",
               "/public/nlp/pretrained_model/realise_well_trained_model.zip"),
    "chinese-electra-180g-small-ex-discriminator":
    Downloader("pretrained_model/chinese-electra-180g-small-ex-discriminator.zip",
               "/public/nlp/pretrained_model/chinese-electra-180g-small-ex-discriminator.zip"),
    "csc_train_v3":
    Downloader("data/csc_train_v3/train_lmdb_1e5.zip",
               "/public/nlp/Chinese Corpus/csc_train_data/data_v3/train_lmdb_1e5.zip"),
    "csc_train_v2":
    Downloader("data/csc_train_v2/csc_dataset_v2.zip",
               "/public/nlp/Chinese Corpus/csc_train_data/data_v2/csc_dataset_v2.zip"),
    "csc_train_v1":
    Downloader("data/csc_train_v1/csc_dataset_v2.zip",
               "/public/nlp/Chinese Corpus/csc_train_data/data_v1/csc_dataset_v1.zip"),

    "chinese_roberta_uer":
    Downloader("pretrained_model/chinese_roberta_uer.zip",
               "/public/nlp/pretrained_model/chinese_roberta_uer.zip"),
    "gpt2_cn_cluesmall":
    Downloader("pretrained_model/gpt2_cn_cluesmall.zip",
               "/public/nlp/pretrained_model/gpt2_cn_cluesmall.zip"),
    "miduCTC_v1_0928":
    Downloader("model/miduCTC_v1_0928.zip",
               "/public/nlp/yjy-gen-corrector/miduCTC_v1_0928.zip"),
    "miduCTC_v1.1_1021":
    Downloader("model/miduCTC_v1.1_1021.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v1.1_1021.zip'),
    "miduCTC_v1.2_1028":
    Downloader("model/miduCTC_v1.2_1028.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v1.2_1028.zip'),
    "miduCTC_v1.3_1104":
    Downloader("model/miduCTC_v1.3_1104.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v1.3_1104.zip'),
    "miduCTC_v1.4_1111":
    Downloader("model/miduCTC_v1.4_1111.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v1.4_1111.zip'),
    "miduCTC_v1.5_1118":
    Downloader("model/miduCTC_v1.5_1118.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v1.5_1118.zip'),
    "miduCTC_v1.6_1125":
    Downloader("model/miduCTC_v1.6_1125.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v1.6_1125.zip'),
    "miduCTC_v1.6_1201":
    Downloader("model/miduCTC_v1.6_1201.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v1.6_1201.zip'),
    "miduCTC_v1.7_1202":
    Downloader("model/miduCTC_v1.7_1202.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v1.7_1202.zip'),
    "miduCTC_v1.8_1209":
    Downloader("model/miduCTC_v1.8_1209.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v1.8_1209.zip'),
    "miduCTC_v1.9_1216":
    Downloader("model/miduCTC_v1.9_1216.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v1.9_1216.zip'),
    "miduCTC_v2.0_1225":
    Downloader("model/miduCTC_v2.0_1225.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v2.0_1225.zip'),
    "miduCTC_v2.1_1230":
    Downloader("model/miduCTC_v2.1_1230.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v2.1_1230.zip'),
    "miduCTC_v2.2_0106":
    Downloader("model/miduCTC_v2.2_0106.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v2.2_0106.zip'),
    "miduCTC_v2.3_0113":
    Downloader("model/miduCTC_v2.3_0113.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v2.3_0113.zip'),
    "miduCTC_v2.3.1_0210":
    Downloader("model/miduCTC_v2.3.1_0210.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v2.3.1_0210.zip'),
    "miduCTC_v2.5.0_0217":
    Downloader("model/miduCTC_v2.5.0_0217.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v2.5.0_0217.zip'),
    "miduCTC_v2.6.0_0224":
    Downloader("model/miduCTC_v2.6.0_0224.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v2.6.0_0224.zip'),
    "miduCTC_v3.2.0_0429":
    Downloader("model/miduCTC_v3.2.0_0429.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v3.2.0_0429.zip'),

    "miduCTC_v3.2.0_0429":
    Downloader("model/miduCTC_v3.2.0_0429.zip",
               '/public/nlp/yjy-gen-corrector/miduCTC_v3.2.0_0429.zip'),


}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select a model you want')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='model name of (roberta, macbert, wwm, electra, electra-large macbert-large, robert-wwm-ext, realise)'
    )
    args = parser.parse_args()

    pretrained_model_dl[args.model].get_path()
