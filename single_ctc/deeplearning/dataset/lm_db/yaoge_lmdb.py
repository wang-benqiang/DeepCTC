

import random

import lmdb
from rich.progress import track
from src import logger
from tqdm import tqdm


class TrainDataLmdb:

    def __init__(self, lmdb_dir, map_size='0g', if_create=False, readonly=True):
        self.path = lmdb_dir
        self.readonly = readonly
        self.number_key = "n_samples"
        self.env, self.correct_db, self.wrong_db = self._create_or_load(
            map_size=map_size, if_create=if_create)
        self.read_txn = self.env.begin(write=False)
        self.correct_cursor = self.read_txn.cursor(db=self.correct_db)
        self.wrong_cursor = self.read_txn.cursor(db=self.wrong_db)

    def __len__(self):

        with self.env.begin(write=False) as txn:
            # crt_cursor = self.read_txn.cursor(db=self.correct_db)
            crt_cursor = txn.cursor(db=self.correct_db)
            length = int(
                str(crt_cursor.get("n_samples".encode()), encoding="utf-8"))
        return length

    def _get_map_size(self, map_size):
        assert map_size[-1].lower() == "g"
        g_times = 10e8
        number = int(map_size[:-1])
        return int(number * g_times)

    def _create_or_load(self, map_size, if_create):
        self.map_size = self._get_map_size(map_size)
        import os
        if not os.path.exists(self.path) and if_create:
            assert map_size != '0g'
            env = lmdb.open(self.path, map_size=self.map_size,
                            max_dbs=6)
            correct_db = env.open_db(b'correct')
            wrong_db = env.open_db(b'wrong')

            with env.begin(write=True) as txn:
                txn.put(key=self.number_key.encode(),
                        value=str(0).encode(), db=correct_db)
            return env, correct_db, wrong_db
        elif os.path.exists(self.path) and not if_create:
            env = lmdb.open(self.path, max_dbs=6,
                            map_size=self.map_size, readonly=self.readonly)
            correct_db = env.open_db(b'correct')
            wrong_db = env.open_db(b'wrong')
            return env, correct_db, wrong_db
        else:
            raise Exception(
                "database not found! or database exists but if_create is True")

    def put_batch_data(self, data_pair):
        # assert len(text_list) == len(seg_list) == len(pos_list)
        with self.env.begin(write=True) as txn:
            sample_cursor = txn.cursor(db=self.correct_db)
            n_samples = int(str(sample_cursor.get(
                self.number_key.encode()), encoding="utf-8"))
            for idx, pair in enumerate(tqdm(data_pair, "writing batch data to train_lmdb")):
                crt, wrg = pair[0], pair[1]
                txn.put(key=str(n_samples).encode(),
                        value=crt.encode(), db=self.correct_db)
                txn.put(key=str(n_samples).encode(),
                        value=wrg.encode(), db=self.wrong_db)
                n_samples += 1
            txn.put(key=self.number_key.encode(), value=str(
                n_samples).encode(), db=self.correct_db)

    def append_pos_samples(self, pos_data_ratio=0.2):
        all_data_len = len(self)
        pos_data_len = int(all_data_len * pos_data_ratio)
        trg_src_paris = []

        if pos_data_len > 0:
            print('read data')
            for i in tqdm(range(pos_data_len), total=pos_data_len):
                src, trg = self.get_src_trg(i)
                trg_src_paris.append((trg, trg))
        print('push data start:')
        self.put_batch_data(trg_src_paris)

    def get_src_trg(self, idx):
        crt = str(self.correct_cursor.get(str(idx).encode()), encoding="utf-8")
        wrg = str(self.wrong_cursor.get(str(idx).encode()), encoding="utf-8")
        return wrg, crt


if __name__ == '__main__':

    """

    ------------
    src:前不久，中国人大网公布了《刑法中华人民共和国修正案（十一）（草案二次审议稿）》。
    trg:前不久，中国人大网公布了《中华人民共和国刑法修正案（十一）（草案二次审议稿）》。
    ------------
    src:这的透露信号再明确不过：国家捍卫高考公平，将有越来越严格的法律手段。
    trg:这透露的信号再明确不过：国家捍卫高考公平，将有越来越严格的法律手段。
    ------------
    src:上大学冒名顶替，严重损害他人利益，破坏教育公平和社会公平正义底线。
    trg:冒名顶替上大学，严重损害他人利益，破坏教育公平和社会公平正义底线。
    ------------

    """

    import os

    sub_dirs = [
        # 'data/csc_train_10e/yaoge/train_csc_lmdb_0120',
        # 'data/csc_train_10e/yaoge/train_csc_lmdb_0123',
        # 'data/csc_train_10e/yaoge/train_csc_lmdb_0124_1',
        # 'data/csc_train_10e/yaoge/train_csc_lmdb_0124_2',
        # # 'data/csc_train_10e/past_data',
        # 'data/csc_train_10e/yaoge/train_csc_lmdb_0124_3',
        # 'data/csc_train_10e/yaoge/train_csc_lmdb_0124_4',
        # 'data/csc_train_10e/yaoge/train_csc_lmdb_0124_5',
        # 'data/csc_train_10e/yaoge/train_lmdb_1e5',
        # 'data/train_lmdb_0228_13e/train_csc_lmdb_0228_1',
        # 'data/train_lmdb_0228_13e/train_csc_lmdb_0228_2',
        # 'data/train_lmdb_0228_13e/train_csc_lmdb_0228_3',
        # 'data/train_lmdb_0228_13e/train_csc_lmdb_0228_4',
        # 'data/train_lmdb_0228_13e/train_csc_lmdb_0228_5',
        # 'data/train_lmdb_0228_13e/train_csc_lmdb_0228_6',
        # 'data/yaoge_train_0314_13e/train_csc_0314_1',
        # 'data/yaoge_train_0314_13e/train_csc_0314_2',
        # 'data/yaoge_train_0314_13e/train_csc_0314_3',
        # 'data/yaoge_train_0314_13e/train_csc_0314_4',
        # 'data/yaoge_train_0314_13e/train_csc_0314_5',
        # 'data/yaoge_train_0314_13e/train_csc_0314_6',
        # 'data/yaoge_train_0314_13e/train_csc_0314_7',
        # 'data/yaoge_train_0314_13e/train_csc_0314_8',
        'data/train_yaoge_lmdb_0228_13e/train_csc_lmdb_0228_1',
 
    ]
# all_len:1953179427
# 
    all_len = 0
    for sub_dir in sub_dirs:
        logger.info('current dir:{}'.format(sub_dir))
        db_mng = TrainDataLmdb(lmdb_dir=sub_dir, map_size='400g', readonly=True)
        # db_mng.append_pos_samples(pos_data_ratio=0.25)
        print('sub_dir:{}, length:{}'.format(sub_dir, len(db_mng)))
        all_len += len(db_mng)
    print('all_len:{}'.format(all_len))
