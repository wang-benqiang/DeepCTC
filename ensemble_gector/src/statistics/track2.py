import json
from difflib import SequenceMatcher


def check_pos_neg_ratio(fp):
    data_list = json.load(open(fp, 'r', encoding='utf8'))
    if 'src' in data_list[0]:
        src_key, trg_key = 'src', 'trg'
    elif 'source' in data_list[0]:
        src_key, trg_key = 'source', 'target'
    pos_num_list = [1 if data[src_key] == data[trg_key] else 0 for data in data_list]
    pos_num = sum(pos_num_list)
    neg_num = len(pos_num_list) - sum(pos_num_list) 
    pos_neg_ratio = round(pos_num/neg_num, 5)
    print('pos_num', pos_num)
    print('neg_num', neg_num)
    print('pos_neg_ratio', pos_neg_ratio)
if __name__ == '__main__':
    fp = 'data/ccl_2022/track2_train/lang8/lang8_train_ccl2022_all.json'
    fp = 'data/ccl_2022/track2_train/cged/all_cged_data.json'
    # fp = 'data/ccl_2022/track2_train/cged_sighan/sighan_train_ccl2022_csc.json'
    check_pos_neg_ratio(fp)