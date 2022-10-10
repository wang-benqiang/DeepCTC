

import os
import random

import lmdb
from tqdm import tqdm
from src import logger


class CtcDBManager:
    def __init__(self,
                 lmdb_dir=None,
                 db_gb_size=None
                 ):
        """used for ctc, save src texts and trg texts

        keys in db:
            src_id
            trg_id
            data_len

        Args:
            lmdb_dir ([type]): [description]
        """
        
        self.lmdb_dir = lmdb_dir
        if lmdb_dir is not None and os.path.exists(lmdb_dir):
            # load

            if db_gb_size is not None:
                db_gb_size = (1024**3) * db_gb_size
                self.db = lmdb.open(lmdb_dir, db_gb_size)
            else:
                self.db = lmdb.open(lmdb_dir)
            self.read_cursor = self.db.begin(write=False)

            logger.info('dataset db loaded from {}'.format(lmdb_dir))
            # logger.info('dataset len {}'.format(self.search('data_len')))
            # need to use create_db to init
        
        else:
            self.init(db_gb_size=db_gb_size)

    def init(self, db_gb_size):
        assert not os.path.exists(self.lmdb_dir), 'lmdb exists!'
        logger.info('start creating db at {}'.format(self.lmdb_dir))
        db_gb_size = (1024**3) * db_gb_size
        logger.info('db size {}'.format(db_gb_size))
        self.db = lmdb.open(self.lmdb_dir, db_gb_size)
        self.write_cursor = self.db.begin(write=True)
        self.insert('data_len', 0)
        self.write_cursor.commit()
        self.read_cursor = self.db.begin(write=False)
        logger.info('db created at {}'.format(self.lmdb_dir))
    
    # def create_db(self,
    #               srcs,
    #               trgs,
    #               db_gb_size=32):
    #     assert not os.path.exists(self.lmdb_dir), 'lmdb exists!'
    #     logger.info('start creating db at {}'.format(self.lmdb_dir))
    #     db_gb_size = (1024**3) * db_gb_size
    #     logger.info('db size {}'.format(db_gb_size))
    #     self.db = lmdb.open(self.lmdb_dir, db_gb_size)
    #     self.write_cursor = self.db.begin(write=True)
    #     logger.info('src_len:{}'.format(len(srcs)))
    #     logger.info('trg_len:{}'.format(len(trgs)))
    #     for i in track(range(len(srcs)), description='Insert data.', total=len(srcs)):
    #         self.insert('src_{}'.format(i), srcs[i])
    #         self.insert('trg_{}'.format(i), trgs[i])
    #     self.insert('data_len', len(srcs))
    #     self.write_cursor.commit()
    #     self.read_cursor = self.db.begin(write=False)
    #     logger.info('db created at {}'.format(self.lmdb_dir))

    def insert(self, k, v):
        self.write_cursor.put(str(k).encode(), str(v).encode())

    def delete(self, k):
        self.write_cursor.delete(str(k).encode())

    
    def append_srcs_trgs(self,srcs, trgs):
        assert len(srcs) == len(trgs)
        data_len = int(self.search('data_len'))
        
        self.write_cursor = self.db.begin(write=True)
        for i in tqdm(range(len(trgs)), total=len(trgs)):
            self.insert('src_{}'.format(i+data_len), srcs[i])
            self.insert('trg_{}'.format(i+data_len), trgs[i])
            
        new_data_len = data_len + len(trgs)
        self.insert('data_len', new_data_len)
        self.write_cursor.commit()
        self.read_cursor = self.db.begin(write=False)
        logger.info('addition_data_len:{}, '.format(len(trgs)))
        logger.info('old_data_len: {}, new_data_len:{}, '.format(
            data_len, new_data_len))
        
        
    def append_pos_samples(self, pos_ratio=0.4, random_mode=True):
        data_len = int(self.search('data_len'))

        pos_data_num = int(data_len * pos_ratio)
        pos_data_num = 1 if pos_data_num < 1 else pos_data_num
        print('pos_data_num:{}'.format(pos_data_num))
        trgs = []
        if random_mode:
            print('random samples index start')
            pos_sample_idx = random.sample(list(range(data_len)), pos_data_num)
            print('random samples index end')
            print('pos_sample_idx:', pos_sample_idx[:10])
            
            for i in tqdm(pos_sample_idx, desc='Read data.', total=len(pos_sample_idx)):
                src, trg =  self.get_src_trg(i)
                if src!=trg:
                    # 只对负样本增加正样本
                    trgs.append(self.get_src_trg(i)[1])
        else:
            for i in tqdm(range(0, pos_data_num), desc='Read data.', total=pos_data_num):
                src, trg =  self.get_src_trg(i)
                if src!=trg:
                    trgs.append(self.get_src_trg(i)[1])
        pos_data_num = len(trgs)
        print('pos trgs[:3]:', trgs[:3])
        self.write_cursor = self.db.begin(write=True)
        for i in tqdm(range(len(trgs)), desc='Insert data.', total=len(trgs)):
            self.insert('src_{}'.format(i+data_len), trgs[i])
            self.insert('trg_{}'.format(i+data_len), trgs[i])

        new_data_len = data_len + pos_data_num
        self.insert('data_len', new_data_len)
        self.write_cursor.commit()
        self.read_cursor = self.db.begin(write=False)
        self.write_cursor = None

        logger.info('pos_data_len:{}, '.format(pos_data_num))
        logger.info('old_data_len: {}, new_data_len:{}, '.format(
            data_len, new_data_len))

    def update(self, k, v):
        self.write_cursor.put(str(k).encode(), str(v).encode())

    def search(self, k):
        v = str(self.read_cursor.get(str(k).encode()), encoding='utf-8')
        return v

    def __len__(self):
        return int(self.search('data_len'))

    def display(self):
        cur = self.read_cursor.cursor()
        for k, v in cur:
            v = str(v, encoding='utf-8')
            print(k, v)

    def get_src_trg(self, idx):
        src, trg = self.search('src_{}'.format(
            idx)), self.search('trg_{}'.format(idx))
        return src, trg


if __name__ == '__main__':
    data_dir = 'data/01_Politics/csc'
    lmdb_dirs = [
        # 'data/gec_train_10e/train_gec_lmdb_0120',
        # 'data/gec_train_10e/train_gec_lmdb_0123',
        # 'data/gec_train_10e/train_gec_lmdb_0124_1',
        # 'data/gec_train_10e/train_gec_lmdb_0124_2',
        # 'data/gec_train_10e/train_gec_lmdb_0124_3',
        # 'data/gec_train_10e/train_gec_lmdb_0124_4',
        # 'data/gec_train_10e/train_gec_lmdb_0124_5',
        # 'data/gec_train_10e/train_lmdb_1e5',
        # 'data/csc_train_10e/past_data', #  
        # 'data/corpus_bj_lmdb/2000w_220611',
        # 'data/corpus_bj_lmdb/word_level_2500w',
        'data/corpus_bj_lmdb/3000w_220611'
        
        # 'data/gec_train_20e/train_gec_lmdb_0207_1',
        # 'data/gec_train_20e/train_gec_lmdb_0207_2',
        # 'data/gec_train_20e/train_gec_lmdb_0207_3',
        # 'data/gec_train_20e/train_gec_lmdb_0207_4',
        # 'data/gec_train_20e/train_gec_lmdb_0221_1',
        # 'data/gec_train_20e/train_gec_lmdb_0221_2',
        # 'data/gec_train_20e/train_gec_lmdb_0221_3',
        # 'data/gec_train_20e/train_gec_lmdb_0221_4',
        # 'db/realise_test'
    ]
    lens = []
    for lmdb_dir in lmdb_dirs:
        db_mng = CtcDBManager(lmdb_dir)
        lens.append(len(db_mng))
        print(len(db_mng))

    print('total lens:{}'.format(sum(lens)))
