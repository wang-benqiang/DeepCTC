
import json
from copy import deepcopy

from cv2 import log
from LAC import LAC
from logs import logger
from src.metrics.ccl_track1 import metric_file
from src.metrics.ccl_track2 import gen_track2_outfile
from src.metrics.metric_csc import f1_csc
from src.predictor.predictor_csc_bert import PredictorCscBert
from src.predictor.predictor_csc_gpt import PredictorGpt
from src.predictor.predictor_csc_mlm import PredictorCscMlm
from src.predictor.predictor_csc_realise import PredictorCscRealise
from src.predictor.predictor_csc_s2e_pronounce import PredictorCscS2ePronounce
from src.predictor.predictor_csc_seq2edit import PredictorCscSeq2Edit
from predictor.predictor_ctc_t5 import PredictorCtcT5
from src.predictor.predictor_csc_t5 import PredictorCscT5
from utils.data_helper import (QUOTATION_CONTENT_RE, replace_char,
                               tradition_to_simple)
from utils.sound_shape_code.ssc import CharHelper



class Corrector:
    def __init__(self,
                 in_model_dir1,
                 in_model_dir2,
                 in_model_dir3,
                 in_model_dir4,
                 #  in_model_dir_gpt,
                 use_cuda=False
                 ):

        # self.predictor1 = PredictorCscRealise(
        #     in_model_dir=in_model_dir1, use_cuda=use_cuda)
        # self.predictor2 = PredictorCscSeq2Edit(
        #     in_model_dir2, use_cuda=use_cuda)
        # self.predictor3 = PredictorCscSeq2Edit(
        #     in_model_dir3, use_cuda=use_cuda)
        self.predictor4 = PredictorCtcT5(
            in_model_dir4, use_cuda=use_cuda)
        # self.predictor_gpt = PredictorGpt(in_model_dir_gpt, use_cuda=use_cuda)
        self.char_helper = CharHelper()
        self.ner_model = LAC(mode='lac')
        self.ner_pos_list = ['PER']

    def correct(self,):
        pass

    def csc(self,
            texts,
            iteration_times=1):
        "seq2edit"

        "preprogress"
        pred_texts = texts
        pred_texts = [tradition_to_simple(i) for i in texts]
        for iter_time in range(iteration_times):
            "model_1"

            pred_texts = self.predictor1.predict(
                pred_texts, prob_threshold=0.49)

            logger.debug('model 1 pred:{}'.format(pred_texts))
            "model 2"
            pred_texts = self.predictor2.predict(
                pred_texts, prob_threshold=0.49)

            logger.debug('model 2 pred:{}'.format(pred_texts))

            "model 3"
            pred_texts = self.predictor3.predict(
                pred_texts, prob_threshold=0.78)

            logger.debug('model 3 pred:{}'.format(pred_texts))
            "model 4"
            # pred_texts = self.predictor4.predict(pred_texts)

            # logger.debug('model 4 pred:{}'.format(pred_texts))

            # "model 4"
            # pred_texts = self.predictor4.predict(pred_texts, prob_threshold=0.49)

            # logger.debug('model 4 pred:{}'.format(pred_texts))
        "lm start"
        # diff_ids, diff_src_texts, diff_pred_texts = [], [], []
        # for i, (src_text, pred_text) in enumerate(zip(texts, pred_texts)):
        #     if src_text != pred_text:
        #         diff_chars = [(src_char, pred_char)
        #                       for src_char, pred_char in zip(src_text, pred_text) if src_char != pred_char]
        #         is_dedide_case = all([True if src_char in '的地得' and pred_char in '的地得' else False for (
        #             src_char, pred_char) in diff_chars])

        #         if not is_dedide_case:
        #             diff_ids.append(i)
        #             diff_src_texts.append(src_text)
        #             diff_pred_texts.append(pred_text)

        # _, lm_scores = self.predictor_gpt.predict(
        #     diff_src_texts+diff_pred_texts)
        # diff_len = len(diff_ids)
        # src_scores, prd_scores = lm_scores[:diff_len], lm_scores[diff_len:]

        # for idx, diff_src_text, diff_pred_text, src_score, prd_score in zip(diff_ids, diff_src_texts, diff_pred_texts, src_scores, prd_scores):
        #     if src_score - prd_score > 0.02:
        #         logger.info('[src, pred]:{}, [src, pred]:{}'.format(
        #             [diff_src_text, diff_pred_text], [src_score, prd_score]))
        #         pred_texts[idx] = texts[idx]
        "lm end"

        "post-progress"
        # pred_texts = self.post_progress([tradition_to_simple(i) for i in texts], pred_texts)

        return pred_texts

    def gec(self, texts):
        pred_texts = self.predictor4.predict(texts)
        logger.debug('model 4 pred:{}'.format(pred_texts))
        return pred_texts

    def get_lm_score(self, texts):
        _, scores = self.predictor_gpt.predict(texts)
        return scores

    def infer_ccl_tack1_test(self,
                             in_fp='tests/data/ccl2022_cltc/yaclc-csc_test.src',
                             out_fp='logs/yaclc-csc-test.lbl',
                             iteration_times=1
                             ):
        src_data = [line.strip().split('\t')
                    for line in open(in_fp, 'r', encoding='utf8')]
        src_texts = [i[1] for i in src_data]
        pred_texts = self.csc(src_texts, iteration_times=iteration_times)

        "log"
        data_file_name = in_fp.split('/')[-1].split('.')[0]
        diff_texts = [{'idx': idx+1, 'src': src_text, 'prd': pred_text, 'type': 'diff 'if src_text != pred_text else 'same'} for idx, (src_text,
                      pred_text) in enumerate(zip(src_texts, pred_texts))]
        out_log = open('logs/{}_predwitht5.json'.format(data_file_name),
                       'w', encoding='utf8')
        json.dump(diff_texts, out_log, ensure_ascii=False, indent=4)

        "output file"

        split_sign = ', '
        out_fp = open(out_fp, 'w', encoding='utf8')
        for src, pred_text in zip(src_data, pred_texts):
            line = src[0]
            src_text = src[1]
            if src_text == pred_text:
                line += (split_sign+'0')
            else:
                for idx, (s, p) in enumerate(zip(src_text, pred_text)):
                    if s != p:
                        idx += 1
                        line += (split_sign+str(idx))
                        line += (split_sign+p)

            out_fp.write(line+'\n')
        out_fp.close()
        return pred_texts

    def infer_ccl_tack2_test(self,
                             in_fp='tests/data/ccl2022_cltc/yaclc-csc_test.src',
                             out_fp='logs/yaclc-csc-test.lbl',
                             out_res_fp='logs/track2-test.txt',
                             iteration_times=1
                             ):
        src_data = [line.strip().split('\t')
                    for line in open(in_fp, 'r', encoding='utf8')]
        src_texts = [i[1] for i in src_data]
        src_ids = [i[0] for i in src_data]
        pred_texts = src_texts
        for i in range(iteration_times):
            pred_texts = self.gec(pred_texts)
        opened_out_fp = open(out_fp, 'w', encoding='utf8')
        for src_id, pred_text in zip(src_ids, pred_texts):
            opened_out_fp.write('{}\t{}\n'.format(src_id, pred_text))

        opened_out_fp.close()

        gen_track2_outfile(in_fp, out_fp, out_res_fp)

        

    def evaluate(self,
                 in_fp='tests/data/ccl2022_cltc/yaclc-csc_dev.src',
                 dev_gold_fp='tests/data/ccl2022_cltc/yaclc-csc_dev.lbl',
                 out_fp='logs/yaclc-csc_dev.lbl.pred',
                 iteration_times=3):
        "evaluating dev set"
        pred_texts = self.infer_ccl_tack1_test(
            in_fp, out_fp, iteration_times=iteration_times)
        mertrics = metric_file(out_fp, dev_gold_fp)
        print(mertrics['Correction'])

        # log error prediction

        data_json = json.load(
            open('tests/data/ccl2022_cltc/yaclc-csc_dev.json', 'r', encoding='utf8'))
        error_log_file = open(
            'logs/yaclc-csc_dev_error_prediction.json', 'w', encoding='utf8')
        error_predction_list = []

        src_texts, trg_texts = [], []
        for line, prd_text in zip(data_json, pred_texts):
            src_texts.append(line['src'])
            trg_texts.append(line['trg'])

            if line['trg'] == line['src'] and line['trg'] != prd_text:
                # error prediction
                error_predction_list.append({
                    'src': line['src'],
                    'trg': line['trg'],
                    'prd': prd_text,
                    'cls': '正样本误报'
                })

            elif line['trg'] != line['src'] and line['src'] == prd_text:
                # error prediction
                error_predction_list.append({
                    'src': line['src'],
                    'trg': line['trg'],
                    'prd': prd_text,
                    'cls': '负样本漏报'
                })

            elif line['trg'] != line['src'] and line['trg'] != prd_text:
                # error prediction
                error_predction_list.append({
                    'src': line['src'],
                    'trg': line['trg'],
                    'prd': prd_text,
                    'cls': '负样本误报'
                })
        json.dump(error_predction_list, error_log_file,
                  ensure_ascii=False, indent=2)
        logger.info('error_data_list len:{}'.format(len(error_predction_list)))

        "new metric"
        print('-'*22)
        r = f1_csc(src_texts, trg_texts, pred_texts)
        print(r)
        return mertrics['Correction']['F1']

    def post_progress(self, src_texts, pred_texts):

        for idx, (src_text, pred_text) in enumerate(zip(src_texts, pred_texts)):

            # new_pred_text = self.restore_src_text_for_entity(src_text, pred_text)
            # new_pred_text = self.restore_src_text_for_quotation_content(src_text, pred_text)
            # new_pred_text = self.restore_src_text_by_sim(src_text, new_pred_text)
            new_pred_text = self.restore_src_text_by_sim(
                src_text, new_pred_text)

            if new_pred_text != pred_text:
                logger.info('post progress:[src_text, pred_text]:{}'.format(
                    [src_text, pred_text]))

            pred_texts[idx] = new_pred_text

        return pred_texts

    # def restore_src_text_by_sim(self, src_text, pred_text):
    #     for src_char, pred_char in zip(src_text, pred_text):
    #         if src_char !=pred_char and self.

    def restore_src_text_for_entity(self, src_text, pred_text):
        """消除实体误报
        Args:
            src_text ([type]): [description]
            trg_text ([type]): [description]
        Return:
            trg_text 
        """

        ner_res = self.ner_model.run(pred_text)
        pos_seq = [pos for (word, pos) in zip(*ner_res) for char in word]
        pred_text = [src_char if pos_seq[i] in self.ner_pos_list else pred_char for i,
                     (src_char, pred_char) in enumerate(zip(src_text, pred_text))]

        return ''.join(pred_text)

    def restore_src_text_for_quotation_content(self, src_text, pred_text):
        "消除引号里内容误报"
        matched_res = [[m.group(), m.span()]
                       for m in QUOTATION_CONTENT_RE.finditer(src_text)]
        for m in matched_res:
            # m = ['“临死股东大会”', (5, 13)]
            for i in range(m[1][0], m[1][1]):
                if len(m[0]) <= 4:
                    pred_text = replace_char(
                        old_string=pred_text, char=src_text[i], index=i)
        return pred_text


if __name__ == '__main__':
    import logging
    logger.setLevel(logging.INFO)
    # crtor = Corrector(
    #     in_model_dir1='model/realise_ccl2022_2022Y07M24D23H/epoch4,ith_db:0,step75,testf1_99_57%,devf1_99_62%',
    #     # in_model_dir2='model/csc_s2e_finetune3_2022Y07M28D10H/epoch4,step88,testepochf1_0.9868,devepochf1_0.9931',
    #     in_model_dir2='model/csc_s2e_finetune4_2022Y08M07D19H/epoch3,step47,testepochf1_0.9853,devepochf1_0.9898',
    #     in_model_dir3='model/miduCTC_v3.7.0_csc3model/csc3',
    #     in_model_dir4='model/csc_t5_finetune_dev_test2_2022Y08M07D18H/epoch5,step12,testepochf1_0.85,devepochf1_0.6667',
    #     # in_model_dir4='pretrained_model/macbert4csc-base-chinese',
    #     # in_model_dir_gpt='model/gpt2_pretrain_lmdb_0207_1_2022Y07M26D17H/epoch1,step137822,testepochloss_115.86,devepochloss_16.09',
    #     use_cuda=True
    # )
    # model/csc_realise_ccl_test_ft1_merge
    # 74
    # crtor = Corrector(
    #     in_model_dir1='model/csc_realise_ccl_finetune_2022Y08M09D18H/epoch4,step67,testepochf1_0.9938,devepochf1_0.9953',
    #     # in_model_dir2='model/csc_s2e_finetune3_2022Y07M28D10H/epoch4,step88,testepochf1_0.9868,devepochf1_0.9931',
    #     in_model_dir2='model/csc_s2e_ccl_finetune4_2022Y08M09D18H/epoch3,step56,testepochf1_0.9833,devepochf1_0.9975',
    #     in_model_dir3='model/miduCTC_v3.7.0_csc3model/csc3',
    #     in_model_dir4='model/csc_t5_finetune_dev_test2_2022Y08M07D18H/epoch5,step12,testepochf1_0.85,devepochf1_0.6667',
    #     # in_model_dir4='pretrained_model/macbert4csc-base-chinese',
    #     # in_model_dir_gpt='model/gpt2_pretrain_lmdb_0207_1_2022Y07M26D17H/epoch1,step137822,testepochloss_115.86,devepochloss_16.09',
    #     use_cuda=True
    # )
    # 77
    # crtor = Corrector(
    #     in_model_dir1='model/csc_realise_ccl_test_ft2_2022Y08M11D19H/epoch4,step78,testepochf1_0.9929,devepochf1_0.9912',
    #     # in_model_dir2='model/csc_s2e_finetune3_2022Y07M28D10H/epoch4,step88,testepochf1_0.9868,devepochf1_0.9931',
    #     in_model_dir2='model/csc_s2e_ccl_finetune_final2_2022Y08M11D19H/epoch5,step66,testepochf1_0.9824,devepochf1_0.9922',
    #     in_model_dir3='model/miduCTC_v3.7.0_csc3model/csc3',
    #     in_model_dir4='model/csc_t5_finetune_dev_test2_2022Y08M07D18H/epoch5,step12,testepochf1_0.85,devepochf1_0.6667',
    #     # in_model_dir4='pretrained_model/macbert4csc-base-chinese',
    #     # in_model_dir_gpt='model/gpt2_pretrain_lmdb_0207_1_2022Y07M26D17H/epoch1,step137822,testepochloss_115.86,devepochloss_16.09',
    #     use_cuda=True
    # )
    # new
    crtor = Corrector(
        in_model_dir1='model/csc_realise_ccl_test_ft2_2022Y08M12D11H/epoch4,step78,testepochf1_0.9929,devepochf1_0.9912',
        # in_model_dir2='model/csc_s2e_finetune3_2022Y07M28D10H/epoch4,step88,testepochf1_0.9868,devepochf1_0.9931',
        in_model_dir2='model/csc_s2e_ccl_finetune_final2_2022Y08M12D11H/epoch6,step66,testepochf1_0.9824,devepochf1_0.9906',
        in_model_dir3='model/miduCTC_v3.7.0_csc3model/csc3',
        in_model_dir4='model/gec_t5_train_cged_normal5_2022Y08M17D14H/epoch5,step293,testepochf1_0.2167,devepochf1_0.543',
        # in_model_dir4='pretrained_model/macbert4csc-base-chinese',
        # in_model_dir_gpt='model/gpt2_pretrain_lmdb_0207_1_2022Y07M26D17H/epoch1,step137822,testepochloss_115.86,devepochloss_16.09',
        use_cuda=True
    )

    # crtor.evaluate(
    #     in_fp='tests/data/ccl2022_cltc/yaclc-csc_dev.src',
    #     dev_gold_fp='tests/data/ccl2022_cltc/yaclc-csc_dev.lbl',
    #     out_fp='logs/yaclc-csc_dev.lbl.pred',
    #     iteration_times=1
    # )

    # crtor.infer_ccl_tack1_test(
    #     in_fp='tests/data/ccl2022_cltc/yaclc-csc_test.src',
    #     out_fp='logs/yaclc-csc-test.lbl',
    #     iteration_times=1
    # )
    crtor.infer_ccl_tack2_test(
        in_fp = 'tests/data/ccl2022_cltc/track2/cged_test.txt',
        out_fp = 'logs/track2_pred.txt',
        out_res_fp = 'logs/cged.pred.txt'
        
    )
    # r= crtor.csc(['这家咖啡屋我来过几次。'])
    print('end')

    # [-2.9345405101776123, -2.957263708114624]
