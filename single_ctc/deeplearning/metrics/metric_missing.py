from difflib import SequenceMatcher

from src.evaluate.metrics import f1


def match_missing_dict(src_text, trg_text):
    """ 

    Args:
        src ([type]): 我今写代了
        trg ([type]): 我今写代码了

    Returns:
        [type]: {3: '码'}
    """
    r = SequenceMatcher(None, src_text, trg_text)
    diffs = r.get_opcodes()
    match_missing_dict = {}
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if tag == 'insert':
            match_missing_dict[i1-1] = trg_text[j1:j2]
    return match_missing_dict


def compute_label_nums(src_text, trg_text, pred_text, log_error_to_fp=None):
    """[summary]
    Args:
        src_text ([type]): [错误文本]
        trg_text ([type]): [正确文本]
        pred_text ([type]): [预测文本]
        log_error_to_fp:   open(f, 'w')
    Returns:
        [type]: [description]
    """

    pred_num, correct_num, ref_num = 0, 0, 0

    if src_text == trg_text == pred_text:
        return pred_num, correct_num, ref_num

    pred_match_missing_dict = match_missing_dict(src_text, pred_text)
    gold_match_missing_dict = match_missing_dict(src_text, trg_text)

    pred_num = len(pred_match_missing_dict)
    ref_num = len(gold_match_missing_dict)
    detect_num = len(pred_match_missing_dict.keys() &
                     gold_match_missing_dict.keys())

    correct_num = 0
    for idx, chars in pred_match_missing_dict.items():
        if idx not in gold_match_missing_dict.keys():
            # 检错
            if log_error_to_fp is not None:
                log_text = '误报(检错纠错)\t{}\t{}\t{}\t{}\t{}\n'.format(
                    src_text, trg_text, pred_text, chars, idx)
                log_error_to_fp.write(log_text)

        elif chars not in gold_match_missing_dict.values():
            # 检对 纠错
            if log_error_to_fp is not None:
                log_text = '错报(检对纠错)\t{}\t{}\t{}\t{}\t{}\n'.format(
                    src_text, trg_text, pred_text, chars, idx)
                log_error_to_fp.write(log_text)

        else:
            # 检对 纠对
            correct_num += 1

    for idx, chars in gold_match_missing_dict.items():
        if idx not in pred_match_missing_dict.keys():
            if log_error_to_fp is not None:
                log_text = '漏报(漏检)\t{}\t{}\t{}\t{}\t{}\n'.format(
                    src_text, trg_text, pred_text, chars, idx)
                log_error_to_fp.write(log_text)
    return (pred_num, detect_num, correct_num, ref_num)


def f1_missing(src_texts, trg_texts, pred_texts, log_error_to_fp=None):
    """[字级别 拼写错误 检测和纠错的f1计算]

    Args:
        src_texts ([type]): [源文本]
        trg_texts ([type]): [目标文本]
        pred_texts ([type]): [预测文本]
        log_error_to_fp : 文本路径

    Returns:
        [type]: [description]
    """
    if isinstance(src_texts, str):
        src_texts = [src_texts]
    if isinstance(trg_texts, str):
        trg_texts = [trg_texts]
    if isinstance(pred_texts, str):
        pred_texts = [pred_texts]
    lines_length = len(trg_texts)
    assert len(src_texts) == lines_length == len(
        pred_texts), 'keep equal length'
    all_pred_num, all_detect_num, all_correct_num, all_ref_num = 0, 0, 0, 0
    if log_error_to_fp is not None:
        f = open(log_error_to_fp, 'w', encoding='utf-8')
        f.write('type\tsrc_text\ttrg_text\tpred_text\tpred_char\tchar_index\n')
    else:
        f = None
    all_nums = [compute_label_nums(src_texts[i], trg_texts[i], pred_texts[i], f)
                for i in range(lines_length)]
    if log_error_to_fp is not None:
        f.close()
    for i in all_nums:
        all_pred_num += i[0]
        all_detect_num += i[1]
        all_correct_num += i[2]
        all_ref_num += i[3]

    d_precision = round(all_detect_num/all_pred_num,
                        4) if all_pred_num != 0 else 0
    d_recall = round(all_detect_num/all_ref_num, 4) if all_ref_num != 0 else 0
    c_precision = round(all_correct_num/all_pred_num,
                        4) if all_pred_num != 0 else 0
    c_recall = round(all_correct_num/all_ref_num, 4) if all_ref_num != 0 else 0
    d_f1, c_f1 = f1(d_precision, d_recall), f1(c_precision, c_recall)

    # sentence level

    pred_sent_num = sum([1 for s, p in zip(src_texts, pred_texts) if s != p])

    correct_sent_num = sum(
        [1 for s, t in zip(pred_texts, trg_texts) if s == t])

    ref_sent_num = sum([1 for s, t in zip(src_texts, trg_texts) if s != t])

    sent_precision = round(correct_sent_num/pred_sent_num,
                           4) if pred_sent_num != 0 else 0
    sent_recall = round(correct_sent_num/ref_sent_num,
                        4) if ref_sent_num != 0 else 0
    sent_f1 = f1(sent_precision, sent_recall)

    print('====== [Char Level] ======')
    print('d_precsion:{}%, d_recall:{}%, d_f1:{}%'.format(
        d_precision*100, d_recall*100, d_f1*100))
    print('c_precsion:{}%, c_recall:{}%, c_f1:{}%'.format(
        c_precision*100, c_recall*100, c_f1*100))
    print('error_char_num: {}'.format(all_ref_num))
    print('====== [Sentence Level] ======')
    print('precsion:{}%, recall:{}%, f1:{}%'.format(
        sent_precision*100, sent_recall*100, sent_f1*100))
    print('sentences_num: ', len(src_texts))
    print('error_sentences_num: {}'.format(ref_sent_num))
    print('recall sentence num: {}'.format(pred_sent_num))
    print('recall right_sentence num: {}'.format(correct_sent_num))
    return (d_precision, d_recall, d_f1), (c_precision, c_recall, c_f1)


if __name__ == '__main__':
    s = ['我今写代了', '今心情非常不']
    t = ['我今天写代码了', '今天心情非常不错']
    p = ['我今写代码了', '今天心情非常不好']
    r = f1_missing(s, t, p, 'logs/report_miss.log')
    print(r)
