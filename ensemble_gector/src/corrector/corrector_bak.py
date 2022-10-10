
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
from src.predictor.predictor_csc_t5 import PredictorCscT5
from utils.data_helper import (QUOTATION_CONTENT_RE, replace_char,
                               tradition_to_simple)
from utils.sound_shape_code.ssc import CharHelper


class Corrector:
    def __init__(self,

                 in_model_dir,

                 use_cuda=False
                 ):

        self.predictor = PredictorCscMlm(
            in_model_dir, use_cuda=use_cuda)
        self.char_helper = CharHelper()
        self.ner_model = LAC(mode='lac')
        self.ner_pos_list = ['PER']

    def correct(self,):
        pass

    def csc(self,
            texts,
            iteration_times=2):
        "seq2edit"

        "preprogress"
        pred_texts = texts
        # pred_texts = [tradition_to_simple(i) for i in texts]
        for iter_time in range(iteration_times):
            "model_1"

            pred_texts = self.predictor.predict(
                pred_texts, prob_threshold=0.97, batch_size=128)

            "model 4"

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

    def get_lm_score(self, texts):
        _, scores = self.predictor_gpt.predict(texts)
        return scores

    def evaluate(self,
                 in_fp='logs/ensemble/cged_pred_contrast.txt',
                 out_fp='logs/ensemble/cged_pred_contrast.txt.out',
                 iteration_times=3):
        "evaluating dev set"

        srcs = [line.strip() for line in open(in_fp, 'r', encoding='utf8')]
        prds = self.csc(srcs)

        out_fp = open(out_fp, 'w', encoding='utf8')

        for i in prds:
            out_fp.write(i+'\n')

        error_num = sum([1 for i, j in zip(srcs, prds) if i != j])
        print('error_num:', error_num)
        return 1

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


if __name__ == '__main__':
    import logging
    logger.setLevel(logging.INFO)

    crtor = Corrector(
        in_model_dir='pretrained_model/chinese-roberta-wwm-ext',
        # in_model_dir_gpt='model/gpt2_pretrain_lmdb_0207_1_2022Y07M26D17H/epoch1,step137822,testepochloss_115.86,devepochloss_16.09',
        use_cuda=True
    )

    crtor.evaluate(
        in_fp='logs/ensemble/cged_pred_contrast.txt',
        out_fp='logs/ensemble/cged_pred_contrast.txt.out',
    )
    # r= crtor.csc(['这家咖啡屋我来过几次。'])
    print('end')

    # [-2.9345405101776123, -2.957263708114624]
