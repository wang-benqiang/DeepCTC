from difflib import SequenceMatcher

from src import logger
from src.evaluate.metrics import f1
import json

class MetricForCtc:

    def __init__(self) -> None:
        pass

    @staticmethod
    def match_error_dict(src_text,
                         trg_text):
        """ 

        Args:
            src ([type]): 我今写代了
            trg ([type]): 我今写代码了

        Returns:
            [type]: {3: '码'}
        """
        try:
            r = SequenceMatcher(None, src_text, trg_text)
            diffs = r.get_opcodes()
            match_csc_dict, match_ms_dict, match_rdt_list = {}, {}, []

            for diff in diffs:
                tag, i1, i2, j1, j2 = diff
                if tag == 'replace':
                    for i, v in enumerate(range(i1, i2)):
                        match_csc_dict[v] = trg_text[j1+i]

                if tag == 'delete':
                    redundant_length = i2-i1
                    post_i1, post_i2 = i1+redundant_length, i2+redundant_length
                    if src_text[i1:i2] == src_text[post_i1:post_i2]:
                        match_rdt_list.append((post_i1, post_i2))
                    else:
                        match_rdt_list.append((i1, i2))
                if tag == 'insert':
                    match_ms_dict[i1-1] = trg_text[j1:j2]

            match_rdt_list = [
                j for i in match_rdt_list for j in range(i[0], i[1])]

        except Exception as e:
            logger.error(e)
            logger.error('src:{}  trg:{}'.format(src_text, trg_text))
            logger.error(e)
        return match_csc_dict, match_rdt_list, match_ms_dict

    @staticmethod
    def compute_ctc_label_nums(src_texts,
                               trg_texts,
                               pred_texts,
                               csc_error_log_fp=None,
                               rdt_error_log_fp=None,
                               ms_error_log_fp=None):

        assert len(src_texts) == len(trg_texts) == len(
            pred_texts), 'keep equal length'
        csc_label_nums_dict = {'ref_num': 0,
                               'correct_num': 0, 'pred_num': 0, 'detect_num': 0}
        rdt_label_nums_dict = {'ref_num': 0, 'correct_num': 0, 'pred_num': 0}
        ms_label_nums_dict = {'ref_num': 0,
                              'correct_num': 0, 'pred_num': 0, 'detect_num': 0}
        
        csc_error_log_list = []
        for i in range(len(src_texts)):

            pred_match_csc_dict, pred_match_rdt_list, pred_match_ms_dict = MetricForCtc.match_error_dict(
                src_texts[i], pred_texts[i])
            gold_match_csc_dict, gold_match_rdt_list, gold_match_ms_dict = MetricForCtc.match_error_dict(
                src_texts[i], trg_texts[i])

            csc_label_num_res = MetricForCtc.compute_csc_label_nums(
                pred_match_csc_dict,
                gold_match_csc_dict,
                src_texts[i],
                trg_texts[i],
                pred_texts[i],
                error_log_fp=csc_error_log_fp,

            )
            
            csc_label_nums_dict['ref_num'] += csc_label_num_res['ref_num']
            csc_label_nums_dict['correct_num'] += csc_label_num_res['correct_num']
            csc_label_nums_dict['pred_num'] += csc_label_num_res['pred_num']
            csc_label_nums_dict['detect_num'] += csc_label_num_res['detect_num']
            csc_error_log_list.extend(csc_label_num_res['json_log_data_list'])
            
          
            rdt_label_num_res = MetricForCtc.compute_rdt_label_nums(
                pred_match_rdt_list,
                gold_match_rdt_list,
                src_texts[i],
                trg_texts[i],
                pred_texts[i],
                error_log_fp=rdt_error_log_fp
            )

            rdt_label_nums_dict['ref_num'] += rdt_label_num_res['ref_num']
            rdt_label_nums_dict['correct_num'] += rdt_label_num_res['correct_num']
            rdt_label_nums_dict['pred_num'] += rdt_label_num_res['pred_num']
            rdt_label_nums_dict['detect_num'] = rdt_label_num_res['detect_num']

            ms_label_nums_res = MetricForCtc.compute_ms_label_nums(
                pred_match_ms_dict,
                gold_match_ms_dict,
                src_texts[i],
                trg_texts[i],
                pred_texts[i],
                error_log_fp=ms_error_log_fp
            )

            ms_label_nums_dict['ref_num'] += ms_label_nums_res['ref_num']
            ms_label_nums_dict['correct_num'] += ms_label_nums_res['correct_num']
            ms_label_nums_dict['pred_num'] += ms_label_nums_res['pred_num']
            ms_label_nums_dict['detect_num'] += ms_label_nums_res['detect_num']

        
        if csc_error_log_fp is not None:
            json.dump(csc_error_log_list, csc_error_log_fp, ensure_ascii=False, indent=2)
            
        return {
            'csc': csc_label_nums_dict,
            'rdt': rdt_label_nums_dict,
            'ms': ms_label_nums_dict,
        }

    @staticmethod
    def compute_csc_label_nums(pred_match_csc_dict,
                               gold_match_csc_dict,
                               src_text,
                               trg_text,
                               pred_text,
                               error_log_fp=None):
        "log文件是open过的"
        csc_pred_num = len(pred_match_csc_dict)
        csc_ref_num = len(gold_match_csc_dict)
        if isinstance(pred_match_csc_dict, dict):
            csc_detect_num = len(pred_match_csc_dict.keys() &
                                 gold_match_csc_dict.keys())
        else:
            csc_detect_num = len(
                set(pred_match_csc_dict).intersection(gold_match_csc_dict))
        correct_num = 0

        
        build_log_info = lambda error_type, src_text, trg_text, prd_text, char_position, src_char, trg_char, pred_char: {
            'error_type': error_type,
            'src_text': src_text,
            'trg_text': trg_text,
            'prd_text': prd_text,
            'char_position': char_position,
            'src_char': src_char,
            'trg_char': trg_char,
            'pred_char': pred_char,
            
        }
        
        json_log_data_list = []
        for idx, pred_char in pred_match_csc_dict.items():
            if idx not in gold_match_csc_dict.keys():
                # 检错
                if error_log_fp is not None:
                    
                    loginfo = build_log_info(
                        error_type='误报(检错纠错)',
                        src_text=src_text,
                        trg_text=trg_text,
                        prd_text=pred_text,
                        char_position=idx,
                        src_char=src_text[idx],
                        trg_char=trg_text[idx],
                        pred_char=pred_char,
                    )
                    json_log_data_list.append(loginfo)
                    
            elif pred_char not in gold_match_csc_dict.values():
                # 检对 纠错

                if error_log_fp is not None:
                    
                    
                    loginfo = build_log_info(
                        error_type='错报(检对纠错)',
                        src_text=src_text,
                        trg_text=trg_text,
                        prd_text=pred_text,
                        char_position=idx,
                        src_char=src_text[idx],
                        trg_char=trg_text[idx],
                        pred_char=pred_char,
                    )
                    json_log_data_list.append(loginfo)
            else:
                correct_num += 1
        for idx, trg_char in gold_match_csc_dict.items():
            if idx not in pred_match_csc_dict.keys():
                # 漏报
                if error_log_fp is not None:
                    loginfo = build_log_info(
                        error_type='漏报',
                        src_text=src_text,
                        trg_text=trg_text,
                        prd_text=pred_text,
                        char_position=idx,
                        src_char=src_text[idx],
                        trg_char=trg_char,
                        pred_char=pred_text[idx],
                    )
                    json_log_data_list.append(loginfo)

        
        
        return {
            'ref_num': csc_ref_num,
            'correct_num': correct_num,
            'pred_num': csc_pred_num,
            'detect_num': csc_detect_num,
            'json_log_data_list':json_log_data_list
        }

    @staticmethod
    def compute_rdt_label_nums(pred_match_rdt_list,
                               gold_match_rdt_list,
                               src_text,
                               trg_text,
                               pred_text,
                               error_log_fp=None):

        rdt_pred_num = len(pred_match_rdt_list)
        rdt_ref_num = len(gold_match_rdt_list)
        correct_num = 0

        for i in pred_match_rdt_list:
            if i not in gold_match_rdt_list:
                if error_log_fp is not None:
                    log_text = '误报\t{}\t{}\t{}\t{}\t{}\n'.format(
                        src_text, trg_text, pred_text, src_text[i], i)
                    error_log_fp.write(log_text)
            else:
                correct_num += 1

        for i in gold_match_rdt_list:
            if i not in pred_match_rdt_list:
                if error_log_fp is not None:
                    log_text = '漏报\t{}\t{}\t{}\t{}\t{}\n'.format(
                        src_text, trg_text, pred_text, src_text[i], i)
                    error_log_fp.write(log_text)

        return {
            'ref_num': rdt_ref_num,
            'correct_num': correct_num,
            'pred_num': rdt_pred_num,
            'detect_num': None

        }

    @staticmethod
    def compute_ms_label_nums(pred_match_ms_dict,
                              gold_match_ms_dict,
                              src_text,
                              trg_text,
                              pred_text,
                              error_log_fp=None):

        ms_pred_num = len(pred_match_ms_dict)
        ms_ref_num = len(gold_match_ms_dict)
        ms_detect_num = len(pred_match_ms_dict.keys() &
                            gold_match_ms_dict.keys())

        correct_num = 0

        for idx, chars in pred_match_ms_dict.items():
            if idx not in gold_match_ms_dict.keys():
                # 检错
                if error_log_fp is not None:
                    log_text = '误报(检错纠错)\t{}\t{}\t{}\t{}\t{}\n'.format(
                        src_text, trg_text, pred_text, chars, idx)
                    error_log_fp.write(log_text)
            elif chars not in gold_match_ms_dict.values():
                # 检对 纠错
                if error_log_fp is not None:
                    log_text = '错报(检对纠错)\t{}\t{}\t{}\t{}\t{}\n'.format(
                        src_text, trg_text, pred_text, chars, idx)
                    error_log_fp.write(log_text)
            else:
                # 检对 纠对
                correct_num += 1
        for idx, chars in gold_match_ms_dict.items():
            if idx not in pred_match_ms_dict.keys():
                if error_log_fp is not None:
                    log_text = '漏报(漏检)\t{}\t{}\t{}\t{}\t{}\n'.format(
                        src_text, trg_text, pred_text, chars, idx)
                    error_log_fp.write(log_text)
        return {
            'ref_num': ms_ref_num,
            'correct_num': correct_num,
            'pred_num': ms_pred_num,
            'detect_num': ms_detect_num
        }

    @staticmethod
    def f1_metric(
            src_texts,
            trg_texts,
            pred_texts,
            ref_num,
            correct_num,
            pred_num,
            detect_num=None):

        if detect_num is not None:

            d_precision = round(detect_num/pred_num,
                                4) if pred_num != 0 else 0
            d_recall = round(detect_num/ref_num, 4) if ref_num != 0 else 0

        c_precision = round(correct_num/pred_num,
                            4) if pred_num != 0 else 0
        c_recall = round(correct_num/ref_num, 4) if ref_num != 0 else 0
        if detect_num is not None:
            d_f1 = f1(d_precision, d_recall)
        c_f1 = f1(c_precision, c_recall)

        # sentence level

        pos_ref_sent_num = sum(
            [1 for s, t in zip(src_texts, trg_texts) if s == t])
        pos_recall_sent_num = sum(
            [1 for s, p, t in zip(src_texts, pred_texts, trg_texts) if s == t and p == t])

        pos_pred_sent_num = sum(
            [1 for s, p in zip(src_texts, pred_texts) if s == p])

        pos_sent_precision = pos_recall_sent_num / \
            pos_pred_sent_num if pos_pred_sent_num > 0 else 0
        pos_sent_recall = pos_recall_sent_num / \
            pos_ref_sent_num if pos_ref_sent_num > 0 else 0
        pos_sent_f1 = f1(pos_sent_precision, pos_sent_recall)

        neg_ref_sent_num = sum(
            [1 for s, t in zip(src_texts, trg_texts) if s != t])
        neg_pred_sent_num = sum(
            [1 for s, p in zip(src_texts, pred_texts) if s != p])
        neg_recall_sent_num = sum(
            [1 for s, p, t in zip(src_texts, pred_texts, trg_texts) if s != t and p == t])

        neg_sent_precision = neg_recall_sent_num / \
            neg_pred_sent_num if neg_pred_sent_num > 0 else 0
        neg_sent_recall = neg_recall_sent_num / \
            neg_ref_sent_num if neg_ref_sent_num > 0 else 0
        neg_sent_f1 = f1(neg_sent_precision, neg_sent_recall)

        print('====== [Neg Char Level] ======')
        if detect_num is not None:
            print('d_precsion:{}%, d_recall:{}%, d_f1:{}%'.format(
                d_precision*100, d_recall*100, d_f1*100))
        print('c_precsion:{}%, c_recall:{}%, c_f1:{}%'.format(
            c_precision*100, c_recall*100, c_f1*100))
        print('error_char_num: {}'.format(ref_num))
        print('====== [Pos Sentence Level] ======')
        print('precsion:{}%, recall:{}%, f1:{}%'.format(
            pos_sent_precision*100, pos_sent_recall*100, pos_sent_f1*100))
        print('sentences_num: ', len(src_texts))
        print('pos_ref_sent_num: {}'.format(pos_ref_sent_num))
        print('pos_pred_sent_num: {}'.format(pos_pred_sent_num))
        print('pos_recall_sent_num: {}'.format(pos_recall_sent_num))
        print('====== [Neg Sentence Level] ======')
        print('precsion:{}%, recall:{}%, f1:{}%'.format(
            neg_sent_precision*100, neg_sent_recall*100, neg_sent_f1*100))
        print('sentences_num: ', len(src_texts))
        print('neg_ref_sent_num: {}'.format(neg_ref_sent_num))
        print('neg_pred_sent_num: {}'.format(neg_pred_sent_num))
        print('neg_recall_sent_num: {}'.format(neg_recall_sent_num))
        if pos_sent_f1 > 0 and neg_sent_f1 > 0:
            macro_sent_f1 = (pos_sent_f1 + neg_sent_f1)/2
        else:
            macro_sent_f1 = 0
        print(
            '====== [ Macro-F1 Sentence Level]: {}% ======'.format(macro_sent_f1*100))
        if detect_num is None:
            return (0, 0, 0), (c_precision, c_recall, c_f1)
        return (d_precision, d_recall, d_f1), (c_precision, c_recall, c_f1), (pos_sent_f1, neg_sent_f1, macro_sent_f1)

    @staticmethod
    def run(
            src_texts,
            trg_texts,
            pred_texts,
            task='iter_all',
            csc_error_log_fp=None,
            rdt_error_log_fp=None,
            ms_error_log_fp=None,
    ):
        "针对单个任务计算f1"

        assert task in ('iter_all', 'csc', 'rdt', 'ms')  # 迭代几个任务，错字，多字，少字

        csc_error_log_fp_map = {
            'csc': csc_error_log_fp,
            'rdt': rdt_error_log_fp,
            'ms': ms_error_log_fp,
        }
        for log_task, error_log_fp in csc_error_log_fp_map.items():
            if error_log_fp is not None:
                csc_error_log_fp_map[log_task] = open(
                    error_log_fp, 'w', encoding='utf8')

        label_nums_dict = MetricForCtc.compute_ctc_label_nums(
            src_texts,
            trg_texts,
            pred_texts,
            csc_error_log_fp=csc_error_log_fp_map['csc'],
            rdt_error_log_fp=csc_error_log_fp_map['rdt'],
            ms_error_log_fp=csc_error_log_fp_map['ms'],)

        task_map = {
            'csc': '错字词',
            'rdt': '多字词',
            'ms': '少字词',
        }

        if task == 'iter_all':

            for t, task_label_nums_dict in label_nums_dict.items():

                print('------ [任务类型:{}, 指标如下] ------'.format(task_map[t]))
                (d_precision, d_recall, d_f1), \
                    (c_precision, c_recall, c_f1), (pos_sent_f1, neg_sent_f1, macro_sent_f1) = MetricForCtc.f1_metric(src_texts,
                                                                                                                      trg_texts,
                                                                                                                      pred_texts,
                                                                                                                      ref_num=task_label_nums_dict[
                                                                                                                          'ref_num'],
                                                                                                                      correct_num=task_label_nums_dict[
                                                                                                                          'correct_num'],
                                                                                                                      pred_num=task_label_nums_dict[
                                                                                                                          'pred_num'],
                                                                                                                      detect_num=task_label_nums_dict[
                                                                                                                          'detect_num'],
                                                                                                                      )
                return 1
        else:
            task_label_nums_dict = label_nums_dict[task]
            print('------ [任务类型:{}, 指标如下] ------'.format(task_map[task]))
            (d_precision, d_recall, d_f1), \
                (c_precision, c_recall, c_f1), (pos_sent_f1, neg_sent_f1, macro_sent_f1) = MetricForCtc.f1_metric(src_texts,
                                                                                                                  trg_texts,
                                                                                                                  pred_texts,
                                                                                                                  ref_num=task_label_nums_dict[
                                                                                                                      'ref_num'],
                                                                                                                  correct_num=task_label_nums_dict[
                                                                                                                      'correct_num'],
                                                                                                                  pred_num=task_label_nums_dict[
                                                                                                                      'pred_num'],
                                                                                                                  detect_num=task_label_nums_dict[
                                                                                                                      'detect_num'],
                                                                                                                  )
            return (d_precision, d_recall, d_f1), (c_precision, c_recall, c_f1), (pos_sent_f1, neg_sent_f1, macro_sent_f1)

    @staticmethod
    def ctc_f1_sentence_level(src_texts, pred_texts, trg_texts, log_fp=None):

        # 句子级别
        pos_ref_num, pos_pred_num, pos_recall_num = 0, 0, 0
        neg_ref_num, neg_pred_num, neg_recall_num = 0, 0, 0
        if log_fp is not None:
            log_fp = open(log_fp, 'w', encoding='utf8')
        log_json_data_list = []
        build_log_info = lambda error_type, src_text, trg_text, pred_text: {
            'error_type': error_type,
            'src_text': src_text,
            'trg_text': trg_text,
            'prd_text': pred_text,
            
        }
        for src_text, pred_text, trg_text in zip(src_texts, pred_texts, trg_texts):

            # 先计算pos

            if src_text == pred_text:
                pos_pred_num += 1
            if src_text == trg_text:
                pos_ref_num += 1
            if src_text == trg_text == pred_text:
                pos_recall_num += 1
            # 再计算neg

            if src_text != trg_text:
                neg_ref_num += 1
            if src_text != pred_text:
                neg_pred_num += 1
            if src_text != trg_text and pred_text == trg_text:
                neg_recall_num += 1

            # 统计log
            if log_fp is not None:
                if src_text == trg_text and pred_text != trg_text:
                    desc = '正样本误报'
                    log_info = build_log_info(desc, src_text, trg_text, pred_text)
                    log_json_data_list.append(log_info)
                elif src_text != trg_text and pred_text == src_text:
                    desc = '负样本漏报'
                    log_info = build_log_info(desc, src_text, trg_text, pred_text)
                    log_json_data_list.append(log_info)
                elif src_text != trg_text and src_text != pred_text and pred_text != trg_text:
                    desc = '负样本误报'
                    log_info = build_log_info(desc, src_text, trg_text, pred_text)
                    log_json_data_list.append(log_info)
                
        if log_fp is not None:
            json.dump(log_json_data_list, log_fp, ensure_ascii=False, indent=2)

        pos_sent_precision = pos_recall_num/pos_pred_num if pos_pred_num > 0 else 0
        pos_sent_recall = pos_recall_num/pos_ref_num if pos_ref_num > 0 else 0
        pos_sent_f1 = f1(pos_sent_precision, pos_sent_recall)

        neg_sent_precision = neg_recall_num/neg_pred_num if neg_pred_num > 0 else 0
        neg_sent_recall = neg_recall_num/neg_ref_num if neg_ref_num > 0 else 0
        neg_sent_f1 = f1(neg_sent_precision, neg_sent_recall)

        print('====== [Pos Sentence Level] ======')
        print('precsion:{}%, recall:{}%, f1:{}%'.format(
            pos_sent_precision*100, pos_sent_recall*100, pos_sent_f1*100))
        print('sentences_num: ', len(src_texts))
        print('pos_ref_sent_num: {}'.format(pos_ref_num))
        print('pos_pred_sent_num: {}'.format(pos_pred_num))
        print('pos_recall_sent_num: {}'.format(pos_recall_num))
        print('====== [Neg Sentence Level] ======')
        print('precsion:{}%, recall:{}%, f1:{}%'.format(
            neg_sent_precision*100, neg_sent_recall*100, neg_sent_f1*100))
        print('sentences_num: ', len(src_texts))
        print('neg_ref_sent_num: {}'.format(neg_ref_num))
        print('neg_pred_sent_num: {}'.format(neg_pred_num))
        print('neg_recall_sent_num: {}'.format(neg_recall_num))
        if pos_sent_f1 > 0 or neg_sent_f1 > 0:
            macro_sent_f1 = (pos_sent_f1 + neg_sent_f1)/2
        else:
            macro_sent_f1 = 0
            print(
                '====== [ Macro-F1 Sentence Level]: {}% ======'.format(macro_sent_f1*100))
        return (pos_sent_precision, pos_sent_recall, pos_sent_f1), (neg_sent_precision, neg_sent_recall, neg_sent_f1), macro_sent_f1

    
    @staticmethod
    def ctc_comp_f1_sentence_level(src_texts, pred_texts, trg_texts, log_fp=None):
        "计算负样本的 句子级 纠正级别 F1"
        correct_ref_num, correct_pred_num, correct_recall_num, correct_f1 = 0, 0, 0, 0
        for src_text, pred_text, trg_text in zip(src_texts, pred_texts, trg_texts):
            if src_text != pred_text:
                correct_pred_num += 1
            if src_text != trg_text:
                correct_ref_num +=1
            if src_text!= trg_text and pred_text == trg_text:
                correct_recall_num +=1
                
        if correct_ref_num == 0:
            # 如果文本中全是正样本（没有错误）
            print('文本中未发现错误，无法计算指标，该指标只计算含有错误的样本。')
            return
        
        correct_precision = 0 if correct_recall_num == 0  else correct_recall_num / correct_pred_num
        correct_recall = 0 if correct_recall_num == 0  else correct_recall_num / correct_ref_num
        correct_f1 = f1(correct_precision, correct_recall)
        
        return correct_f1
                
    @staticmethod
    def ctc_comp_f1_token_level(src_texts, pred_texts, trg_texts, log_fp=None):
        "字级别，负样本 检测级别*0.8+纠正级别*0.2 f1"
        def compute_detect_correct_label_list(src_text, trg_text):
            detect_ref_list, correct_ref_list = [], []
            diffs = SequenceMatcher(None, src_text, trg_text).get_opcodes()
            for (tag, src_i1, src_i2, trg_i1, trg_i2) in diffs:

                if tag == 'replace':
                    for count, src_i in enumerate(range(src_i1, src_i2)):
                        trg_token = trg_text[src_i+count]
                        detect_ref_list.append(src_i)
                        correct_ref_list.append((src_i, trg_token))

                elif tag == 'delete':
                    for count, src_i in enumerate(range(src_i1, src_i2)):
                        trg_token = ''
                        detect_ref_list.append(src_i)
                        correct_ref_list.append((src_i, trg_token))

                elif tag == 'insert':
                    trg_token = trg_text[trg_i1:trg_i2]
                    detect_ref_list.append(src_i)
                    correct_ref_list.append((src_i, trg_token))

            return detect_ref_list, correct_ref_list

        # 字级别
        detect_ref_num, detect_pred_num, detect_recall_num, detect_f1 = 0, 0, 0, 0
        correct_ref_num, correct_pred_num, correct_recall_num, correct_f1 = 0, 0, 0, 0

        for src_text, pred_text, trg_text in zip(src_texts, pred_texts, trg_texts):
            # 先统计检测和纠正标签
            detect_pred_list, correct_pred_list = compute_detect_correct_label_list(
                src_text, pred_text)
            detect_ref_list, correct_ref_list = compute_detect_correct_label_list(
                src_text, trg_text)

            detect_ref_num += len(detect_ref_list)
            detect_pred_num += len(detect_pred_list)
            detect_recall_num += len(set(detect_ref_list)
                                     & set(detect_pred_list))

            correct_ref_num += len(correct_ref_list)
            correct_pred_num += len(correct_pred_list)
            correct_recall_num += len(set(correct_ref_list)
                                      & set(correct_pred_list))

        if detect_ref_num == 0:
            # 如果文本中全是正样本（没有错误）
            print('文本中未发现错误，无法计算指标，该指标只计算含有错误的样本。')
            return
        
        detect_precision = 0 if detect_pred_num == 0 else detect_recall_num / detect_pred_num
        detect_recall = 0 if detect_ref_num == 0 else detect_recall_num / detect_ref_num

        correct_precision = 0 if detect_pred_num == 0 else correct_recall_num / correct_pred_num
        correct_recall = 0 if detect_ref_num == 0 else correct_recall_num / correct_ref_num

        detect_f1 = f1(detect_precision, detect_recall)
        correct_f1 = f1(correct_precision, correct_recall)

        final_f1 = detect_f1*0.8+correct_f1*0.2

        return final_f1


if __name__ == '__main__':
    s = ['近天的心情心情不搓，你觉呢', '天气了', '这东西不了', '今天心情心情不错不错']
    t = ['今天的心情不错，你觉得呢', '天晴了', '这个东西不错了', '今天心情不错']
    p = ['今天的心情不搓，你觉得呢', '天晴啊了', '这个东西不好了', '今天心情不错不错']
    r = MetricForCtc.ctc_f1_sentence_level(s, t, p)
    print(r)
    print('end..')
