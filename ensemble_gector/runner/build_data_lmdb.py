import gc
import json
import os

from logs import logger
from tqdm import tqdm
from utils.lmdb.db_manager import CtcDBManager


def build_lmdb(data_dir_list,
               lmdb_dir,
               db_gb_size=20,
               max_dataset_len=None,
               pos_sample_gen_ratio=0.1):
    """读取ctc数据

    Args:
        keep_one_append ([type]): 是否迭代生成多条数据
        chunk_num (int, optional): [description]. Defaults to 100000.

    Returns:
        list:[ [['$START', '$KEEP'], ['明', '$KEEP']], ...]
    """
    all_file_fp = []
    for data_dir in data_dir_list:
        current_dir_files = os.listdir(data_dir)
        print(data_dir)
        print(current_dir_files)
        current_dir_file_fp = [
            '{}/{}'.format(data_dir, f_name) for f_name in current_dir_files
        ]
        all_file_fp.extend(current_dir_file_fp)

    all_src, all_trg = [], []
    print('loading {} files:{}'.format(len(all_file_fp), str(all_file_fp)))
    for file_fp in all_file_fp:
        file_data_len = 0
        print('process file:{}'.format(file_fp))

        if 'txt' in file_fp:
            for line in tqdm(open(file_fp, 'r', encoding='utf-8')):
                line = line.strip().replace(' ', '').split('\t')
                try:
                    if len(line) >= 2:
                        file_data_len += 1
                        all_trg.append(line[0].strip())
                        all_src.append(line[1].strip())
                except:
                    print(line)
            logger.info('file:{}, data len:{}'.format(file_fp, file_data_len))
        elif 'json' in file_fp:
            json_data = json.load(open(file_fp, 'r', encoding='utf8'))
            if not isinstance(json_data, list):
                print('ignore file: {}, continue'.format(file_fp))
                continue
            src_key = 'src' if 'src' in json_data[0] else 'source'
            trg_key = 'trg' if 'trg' in json_data[0] else 'target'

            for item in json_data:
                if item[trg_key] != item[src_key]:
                    all_trg.append(item[trg_key])
                    all_src.append(item[src_key])
        else:
            print('ignore file:{}'.format(file_fp))

        if max_dataset_len is not None:
            if len(all_src) > max_dataset_len:
                all_src = all_src[0:max_dataset_len]
                all_trg = all_trg[0:max_dataset_len]
                return all_src, all_trg
        gc.collect()
    if os.path.exists(lmdb_dir):
        print('lmdb dir:{} exists!'.format(lmdb_dir))
    else:

        lmdb_manager = CtcDBManager(lmdb_dir=lmdb_dir, db_gb_size=db_gb_size)
        lmdb_manager.append_srcs_trgs(all_src, all_trg)
        if pos_sample_gen_ratio > 0:
            lmdb_manager.append_pos_samples(pos_sample_gen_ratio)
    return 1


if __name__ == '__main__':
    # 先生成数据
    build_lmdb([
        # 'data/artificial_data/csc/1206',
        # 'data/artificial_data/csc/1122',
        # 'data/artificial_data/pos/new',
        # 'data/artificial_data/csc/0105',
        # 'data/artificial_data/csc/0117',
        # 'data/artificial_data/csc/0216',
        # 'data/artificial_data/csc/0222',
        # 'data/artificial_data/csc/past',
        # 'data/artificial_data/csc/0307',
        # 'data/artificial_data/csc/new',
        # 'data/artificial_data/csc/new',
        # 'data/artificial_data/pos/1206',
        # 'data/artificial_data/pos/0216',
        # 'data/artificial_data/pos/past_data',
        # 'tests/data/ccl2022_cltc',
        # 'tests/data/ccl2022_cltc',
        # 'data/ccl_2022/track1_train/csc_test',
        # 'data/ccl_2022/track1_train/csc_test',
        # 'data/ccl_2022/track1_train/csc_test',
        # 'data/ccl_2022/track2_train/cged',
        # 'data/ccl_2022/track1_train/csc_train',
        # 'data/ccl_2022/track2_train/lang8',
        # 'data/ccl_2022/track2_train/cfl_data',
        # 'data/ccl_2022/track2_train/cged_sighan',
        'data/ccl_2022/cged/cged2021',
        # 'data/ccl_2022/track2_train/cged'
    ], lmdb_dir='data/lmdb/cged2021+100pos', db_gb_size=2, pos_sample_gen_ratio=1)
