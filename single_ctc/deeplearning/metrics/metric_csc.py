from src.evaluate.metrics import f1


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
    assert len(src_text) == len(trg_text) == len(
        pred_text), 'src_text:{}, trg_text:{}, pred_text:{}'.format(src_text, trg_text, pred_text)
    pred_num, detect_num, correct_num, ref_num = 0, 0, 0, 0

    for j in range(len(trg_text)):
        src_char, trg_char, pred_char = src_text[j], trg_text[j], pred_text[j]
        if src_char != trg_char:
            ref_num += 1
            if src_char != pred_char:
                detect_num += 1
            elif log_error_to_fp is not None and pred_char != trg_char and pred_char == src_char:
                log_text = '漏报\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    src_text, trg_text, src_char, trg_char, pred_char, j)
                log_error_to_fp.write(log_text)

        if src_char != pred_char:
            pred_num += 1
            if pred_char == trg_char:
                correct_num += 1
            elif log_error_to_fp is not None and pred_char != trg_char and src_char == trg_char:
                log_text = '误报\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    src_text, trg_text, src_char, trg_char, pred_char, j)
                log_error_to_fp.write(log_text)
            elif log_error_to_fp is not None and pred_char != trg_char and src_char != trg_char:
                log_text = '错报(检对报错)\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    src_text, trg_text, src_char, trg_char, pred_char, j)
                log_error_to_fp.write(log_text)

    return (pred_num, detect_num, correct_num, ref_num)


def f1_csc(src_texts, trg_texts, pred_texts, log_error_to_fp=None):
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
        f.write('type\tsrc_text\ttrg_text\tsrc_char\ttrg_char\tpred_char\tchar_index\n')
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
    s = ['你号钟国', '心晴不搓']
    t = ['你号钟国', '心晴不搓']
    p = ['你号中国', '心青不搓']
    r = f1_csc(s, t, p, 'logs/report.txt')
    print(r)
