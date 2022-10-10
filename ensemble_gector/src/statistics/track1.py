import json
from difflib import SequenceMatcher


def dev_distribution(data_fp='tests/data/ccl2022_cltc/yaclc-csc_dev.json'):
    data_list = json.load(open(data_fp, 'r', encoding='utf8'))
    nums_pos_samples, nums_neg_samples = 0, 0
    num_error_chars_dict_every_sample = {}
    confusion_dict = {}
    for item in data_list:
        src_text, trg_text = item['src'], item['trg']
        diffs = SequenceMatcher(None, item['src'], item['trg']).get_opcodes()

        for (op_tag, src_i1, src_i2, trg_i1, trg_i2) in diffs:
            if op_tag == 'replace':
                assert src_i2 - src_i1 == trg_i2 - trg_i1
                src_str, trg_str = src_text[src_i1:src_i2], trg_text[trg_i1:trg_i2]
                if src_str in confusion_dict:
                    confusion_dict[src_str].append(trg_str)
                else:
                    confusion_dict[src_str] = [trg_str]

        error_char_list = [[idx, src_char, trg_char] for idx, (src_char, trg_char) in enumerate(
            zip(item['src'], item['trg'])) if src_char != trg_char]

        error_char_nums = len(error_char_list)
        if len(error_char_list) == 0:
            nums_pos_samples += 1
        else:
            nums_neg_samples += 1
            if error_char_nums in num_error_chars_dict_every_sample:
                num_error_chars_dict_every_sample[error_char_nums] += 1
            else:
                num_error_chars_dict_every_sample[error_char_nums] = 1

    for k, v in confusion_dict.items():
        confusion_dict[k] = list(set(v))
    
    print('nums_pos_samples:{}, nums_neg_samples:{}, neg/pos:{}'.format(nums_pos_samples, nums_neg_samples, nums_neg_samples/nums_pos_samples))
    print('num_error_chars_dict_every_sample:{}'.format(num_error_chars_dict_every_sample))
    print('confusion_dict:{}'.format(confusion_dict))
    out_data = {
      'nums_pos_samples':nums_pos_samples,
      'nums_neg_samples':nums_neg_samples,
      'neg/pos':nums_neg_samples/nums_pos_samples,
      'num_error_chars_dict_every_sample':num_error_chars_dict_every_sample,
      'confusion_dict':confusion_dict,
    }
    out_fp = open('tests/data/ccl2022_cltc/dev_distribution.json', 'w', encoding='utf8')
    json.dump(out_data, out_fp, ensure_ascii=False, indent=2)
    
    
def test_distribution(data_fp='data/ccl_2022/track1.txt'):

    nums_pos_samples, nums_neg_samples = 0, 0
    num_error_chars_dict_every_sample = {}
    confusion_dict = {}
    for item in open(data_fp, 'r', encoding='utf8'):
        trg_text, src_text = item.strip().split('\t')
        diffs = SequenceMatcher(None, src_text, trg_text).get_opcodes()

        for (op_tag, src_i1, src_i2, trg_i1, trg_i2) in diffs:
            if op_tag == 'replace':
                assert src_i2 - src_i1 == trg_i2 - trg_i1
                src_str, trg_str = src_text[src_i1:src_i2], trg_text[trg_i1:trg_i2]
                if src_str in confusion_dict:
                    confusion_dict[src_str].append(trg_str)
                else:
                    confusion_dict[src_str] = [trg_str]

        error_char_list = [[idx, src_char, trg_char] for idx, (src_char, trg_char) in enumerate(
            zip(src_text, trg_text)) if src_char != trg_char]

        error_char_nums = len(error_char_list)
        if len(error_char_list) == 0:
            nums_pos_samples += 1
        else:
            nums_neg_samples += 1
            if error_char_nums in num_error_chars_dict_every_sample:
                num_error_chars_dict_every_sample[error_char_nums] += 1
            else:
                num_error_chars_dict_every_sample[error_char_nums] = 1

    for k, v in confusion_dict.items():
        confusion_dict[k] = list(set(v))
    
    print('nums_pos_samples:{}, nums_neg_samples:{}, neg/pos:{}'.format(nums_pos_samples, nums_neg_samples, nums_neg_samples/nums_pos_samples))
    print('num_error_chars_dict_every_sample:{}'.format(num_error_chars_dict_every_sample))
    print('confusion_dict:{}'.format(confusion_dict))
    out_data = {
      'nums_pos_samples':nums_pos_samples,
      'nums_neg_samples':nums_neg_samples,
      'neg/pos':nums_neg_samples/nums_pos_samples,
      'num_error_chars_dict_every_sample':num_error_chars_dict_every_sample,
      'confusion_dict':confusion_dict,
    }
    out_fp = open('tests/data/ccl2022_cltc/test_distribution.json', 'w', encoding='utf8')
    json.dump(out_data, out_fp, ensure_ascii=False, indent=2)


if __name__ == '__main__':
  test_distribution()