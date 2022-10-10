
import json
from copy import deepcopy
from tqdm import tqdm
from LAC import LAC
# from logs import logger
from src.metrics.ccl_track1 import metric_file
from src.metrics.ccl_track2 import gen_track2_outfile
from src.metrics.metric_csc import f1_csc
from src.predictor.predictor_csc_bert import PredictorCscBert
from src.predictor.predictor_csc_gpt import PredictorGpt
from src.predictor.predictor_csc_mlm import PredictorCscMlm
from src.predictor.predictor_csc_realise import PredictorCscRealise
from src.predictor.predictor_csc_s2e_pronounce import PredictorCscS2ePronounce
from src.predictor.predictor_csc_seq2edit import PredictorCscSeq2Edit
from src.predictor.predictor_ctc_t5 import PredictorCtcT5
from src.predictor.predictor_ctc_seq2edit import PredictorCtcSeq2Edit
from src.predictor.predictor_gec_seq2edit import PredictorGecSeq2Edit
from src.predictor.predictor_csc_t5 import PredictorCscT5
from src.predictor.predictor_woe_seq2edit import PredictorWoeSeq2Edit
from utils.data_helper import (QUOTATION_CONTENT_RE, replace_char,
                               tradition_to_simple)
from utils.sound_shape_code.ssc import CharHelper
from src.metrics.metric_csc import f1_csc
from copy import deepcopy
import torch


class Corrector:
    def __init__(self,
                 in_model_dir1,

                 in_model_dir_gec,
                 in_model_dir_ctc,

                 batch_size=128,
                 use_cuda=False,
                 cuda_id=0
                 ):

        self.predictor1 = PredictorCscRealise(
            in_model_dir=in_model_dir1, use_cuda=use_cuda,cuda_id=cuda_id)

        self.predictor_ctc = PredictorCtcSeq2Edit(
            in_model_dir_ctc, use_cuda=use_cuda)
        self.predictor_gec = PredictorGecSeq2Edit(
            in_model_dir_gec, use_cuda=use_cuda)

        self.char_helper = CharHelper()
        self.batch_size = batch_size
        self.ner_model = LAC(mode='lac')
        self.ner_pos_list = ['PER']

    def correct(self,):
        pass

    def gec(self,
            texts,
            iteration_times=2):
        "seq2edit"
        "preprogress"
        pred_texts = texts
        pred_texts = [tradition_to_simple(i) for i in texts]
        print('texts num', len(texts))
        pred_texts = self.predictor1.predict(
            pred_texts, batch_size=self.batch_size, prob_threshold=0.49)
        pred_texts = self.predictor_gec.predict(
            pred_texts, batch_size=self.batch_size, prob_threshold=0.68)
        for iter_time in tqdm(range(iteration_times)):
            "model_1"

            pred_texts = self.predictor_ctc.predict(
                pred_texts, batch_size=self.batch_size, prob_threshold=0)

            # logger.debug('model gec pred:{}'.format(pred_texts))

            "model 4"

            # logger.debug('model 4 pred:{}'.format(pred_texts))

        return pred_texts

    def get_lm_score(self, texts):
        _, scores = self.predictor_gpt.predict(texts)
        return scores

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
        pred_texts = self.gec(pred_texts, iteration_times)
        opened_out_fp = open(out_fp, 'w', encoding='utf8')
        for src_id, pred_text in zip(src_ids, pred_texts):
            opened_out_fp.write('{}\t{}\n'.format(src_id, pred_text))

        opened_out_fp.close()

        gen_track2_outfile(in_fp, out_fp, out_res_fp)

    def evaluate(self,
                 in_fp='data/ccl_2022/cged/cged2021/cged_2021_test.json',
                 iteration_times=1):
        src_data = json.load(open(in_fp, 'r', encoding='utf8'))
        src_texts = [i['source'] for i in src_data]
        trg_texts = [i['target'] for i in src_data]

        pos_texts = [i['target']
                     for i in src_data if i['source'] != i['target']]

        src_texts += pos_texts
        trg_texts += pos_texts

        pred_texts = deepcopy(src_texts)
        pred_texts = self.gec(pred_texts, iteration_times=iteration_times)
        r = f1_csc(src_texts, trg_texts, pred_texts)
        print(r)

    def post_progress(self, src_texts, pred_texts):

        for idx, (src_text, pred_text) in enumerate(zip(src_texts, pred_texts)):

            # new_pred_text = self.restore_src_text_for_entity(src_text, pred_text)
            # new_pred_text = self.restore_src_text_for_quotation_content(src_text, pred_text)
            # new_pred_text = self.restore_src_text_by_sim(src_text, new_pred_text)
            new_pred_text = self.restore_src_text_by_sim(
                src_text, new_pred_text)

            # if new_pred_text != pred_text:
            #     logger.info('post progress:[src_text, pred_text]:{}'.format(
            #         [src_text, pred_text]))

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


import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gector_dir', default='model/cltc2/gector')
    parser.add_argument('--test_fp', default='tests/data/ccl2022_cltc/track2/cged_test.txt')
    parser.add_argument('--out_fp', default='logs/gector/track2_pred_1model_3.txt')
    parser.add_argument('--out_res_fp', default='logs/gector/cged.pred_1model_3.txt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import logging
    #logger.setLevel(logging.INFO)
    args = parse_args()
    print('gector_model:', args.gector_dir)
    print('========= loading models =========')
    crtor = Corrector(
        in_model_dir1='model/cltc2/realise',
        in_model_dir_ctc=args.gector_dir,
        # in_model_dir_ctc='model/gector_finetune_sighan_cged_pos100_2022Y08M22D19H/epoch7,step336,testepochf1_0.3007,devepochf1_0.4527',
        in_model_dir_gec='model/cltc2/gec',
        use_cuda=True,
        batch_size=256,
        cuda_id=3
    )
    print('========= predicting =========')
    crtor.infer_ccl_tack2_test(
        in_fp = args.test_fp,
        out_fp = args.out_fp,
        out_res_fp = args.out_res_fp,
        iteration_times = 1,

    )
    print('end')

