from difflib import SequenceMatcher
from src.evaluate.metrics import f1


"多字没有不需要计算detect score, 直接检测后就删除"


def match_redundant_idx(src_text, trg_text):
    """返回需要删除的索引范围, 叠字叠词删除靠后的索引

    Args:
        src ([type]): 今天天吃饭吃饭了
        trg ([type]): 今天吃饭了

    Returns:
        [type]: [2, 5, 6]
    """

    r = SequenceMatcher(None, src_text, trg_text)
    diffs = r.get_opcodes()
    redundant_range_list = []
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if tag == 'delete':
            redundant_length = i2-i1
            post_i1, post_i2 = i1+redundant_length, i2+redundant_length
            if src_text[i1:i2] == src_text[post_i1:post_i2]:
                redundant_range_list.append((post_i1, post_i2))
            else:
                redundant_range_list.append((i1, i2))
    redundant_ids_list = [
        j for i in redundant_range_list for j in range(i[0], i[1])]

    return redundant_ids_list


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

    pred_redundant_ids = match_redundant_idx(src_text, pred_text)
    gold_redundant_ids = match_redundant_idx(src_text, trg_text)

    pred_num = len(pred_redundant_ids)
    ref_num = len(gold_redundant_ids)
    correct_num = 0
    for i in pred_redundant_ids:
        if i not in gold_redundant_ids:
            if log_error_to_fp is not None:
                log_text = '误报\t{}\t{}\t{}\t{}\t{}\n'.format(
                    src_text, trg_text, pred_text, src_text[i], i)
                log_error_to_fp.write(log_text)
        else:
            correct_num += 1

    for i in gold_redundant_ids:
        if i not in pred_redundant_ids:
            if log_error_to_fp is not None:
                log_text = '漏报\t{}\t{}\t{}\t{}\t{}\n'.format(
                    src_text, trg_text, pred_text, src_text[i], i)
                log_error_to_fp.write(log_text)
    return (pred_num, correct_num, ref_num)


def f1_redundant(src_texts, trg_texts, pred_texts, log_error_to_fp=None):
    """[字级别 少字错误 检测和纠错的f1计算]

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
    all_pred_num, all_correct_num, all_ref_num = 0, 0, 0
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
        all_correct_num += i[1]
        all_ref_num += i[2]

    c_precision = round(all_correct_num/all_pred_num,
                        4) if all_pred_num != 0 else 0
    c_recall = round(all_correct_num/all_ref_num, 4) if all_ref_num != 0 else 0
    c_f1 = f1(c_precision, c_recall)

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
    return (0,0,0), (c_precision, c_recall, c_f1)


if __name__ == '__main__':
    s = ['今天天吃饭吃饭了', '今天的心情情不错错']
    t = ['今天吃饭了', '今天心情不错']
    p = ['今天吃饭了', '今天心情不错']
    r = f1_redundant(s, t, p, None)
    print(r)
