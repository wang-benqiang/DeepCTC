from src.metrics.f1_score import f1


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
    """[句子级别f1计算]

    Args:
        src_texts ([type]): [源文本]
        trg_texts ([type]): [目标文本]
        pred_texts ([type]): [预测文本]
        log_error_to_fp : 文本路径

    Returns:
        [type]: [description]
    """

    # sentence level

    pred_sent_num = sum([1 for s, p in zip(src_texts, pred_texts) if s != p])

    correct_sent_num = sum(
        [1 for s, p, t in zip(src_texts, pred_texts, trg_texts) if p == t and s != p])

    ref_sent_num = sum([1 for s, t in zip(src_texts, trg_texts) if s != t])

    sent_precision = round(correct_sent_num/pred_sent_num,
                           5) if pred_sent_num != 0 else 0
    sent_recall = round(correct_sent_num/ref_sent_num,
                        5) if ref_sent_num != 0 else 0
    sent_f1 = f1(sent_precision, sent_recall)

    # print('====== [Sentence Level] ======')
    # print('precsion:{}%, recall:{}%, f1:{}%'.format(
    #     sent_precision*100, sent_recall*100, sent_f1*100))
    # print('sentences_num: ', len(src_texts))
    # print('error_sentences_num: {}'.format(ref_sent_num))
    # print('recall sentence num: {}'.format(pred_sent_num))
    # print('recall right_sentence num: {}'.format(correct_sent_num))
    return (sent_precision, sent_recall, sent_f1)


if __name__ == '__main__':
    s = ['你号钟国', '心晴不搓']
    t = ['你号钟国', '心晴不搓']
    p = ['你号中国', '心青不搓']
    r = f1_csc(s, t, p, 'logs/report.txt')
    print(r)
