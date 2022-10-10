import json
import os
import numpy as np
# from single_ctc.predictor.predictor_csc_realise import PredictorCscRealise
# from single_ctc.predictor.predictor_woe_seq2edit import PredictorWoeSeq2Edit
# from single_ctc.predictor.predictor_gec_seq2edit import PredictorGecSeq2Edit
# from single_ctc.predictor.predictor_csc_bert import PredictorCscBert


def ctc_ensemble(texts):
    save_path = '../single_ctc/single_prediction/result_single.json'
    if os.path.exists(save_path):
        texts,true_result_ids = json.load(open(save_path,'r',encoding='utf-8'))
    else:
        cuda_id = 0
        woe = PredictorWoeSeq2Edit(
            in_model_dir='./model/woe',
            use_cuda=True, cuda_id=cuda_id)
        r = woe.predict(texts, prob_threshold=0.8)

        csc = PredictorCscRealise(
            in_model_dir='./model/csc',
            use_cuda=True, cuda_id=cuda_id)
        r = csc.predict(r, return_topk=5, prob_threshold=0.8)

        csc2 = PredictorCscRealise(
            in_model_dir='./model/csc2',
            use_cuda=True, cuda_id=cuda_id)
        r = csc2.predict(r, return_topk=5, prob_threshold=0.8)

        csc3 = PredictorCscBert(
            in_model_dir='./model/csc3',
            use_cuda=True, cuda_id=cuda_id)
        r = csc3.predict(r, return_topk=5, prob_threshold=0.8)
        # print(r)
        gec = PredictorGecSeq2Edit(
            in_model_dir='./model/gec',
            use_cuda=True, cuda_id=cuda_id)
        r = gec.predict(r, prob_threshold=0.9)
        true_result_ids = list(np.where(np.array(r) == np.array(texts))[0])
        print(len(true_result_ids))
        json.dump((r, true_result_ids), open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    return texts,true_result_ids