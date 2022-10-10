
from ner_ctc_predict import ner_ctc_predict,mixed_predict
from single import ctc_all,single_ensemble
import logging
from ctc_cls.cls_prediction import cls_correction,SeqClsModel
from zhconv import convert
import json
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def load_cged_test_data():
    with open('cged_test.txt','r',encoding='utf-8') as f:
        data=f.readlines()
        data=[i.split('\t')[1].strip() for i in data]
    return data

def data_util(cls_result,ner_ctc_texts,midu_true_result_ids,cged_test_data,ner_cls_result,midu_texts):
    assert len(cls_result) == len(cged_test_data)
    print("总数",len(cged_test_data))
    result = []
    true_number=0
    for ids in range(len(cls_result)):
        single_class = cls_result[ids][0]
        single_logit = cls_result[ids][1]

        ner_single_class = ner_cls_result[ids][0]
        ner_single_logit = ner_cls_result[ids][1]


        ner_ctc_text = ner_ctc_texts[ids]
        cged_data = cged_test_data[ids]
        if single_class == 0:
            if ids not in midu_true_result_ids:
                if ner_single_class==0 and single_logit-ner_single_logit>0.1:
                    result.append(cged_data)
                    true_number += 1
                else:
                    result.append(ner_ctc_text)

            else:
                result.append(cged_data)
                true_number+=1
        else:
            result.append(ner_ctc_text)
    return result

def main():
    logger.info("开始读取数据")
    cged_test_data = load_cged_test_data()
    logger.info("繁体转简体")
    cged_test_data = [convert(i, 'zh-cn') for i in cged_test_data]

    logger.info("数据读取完毕")

    logger.info("开始访问midu")
    single_texts,single_true_result_ids=single_ensemble(cged_test_data)


    logger.info("访问完毕")

    logger.info("开始gector")
    ner_ctc_texts=mixed_predict(single_texts,cged_test_data)
    logger.info("gector处理完毕")

    logger.info("开始分类")
    cls_result = cls_correction(cged_test_data)
    ner_cls_result = cls_correction(ner_ctc_texts)
    logger.info("分类完毕")


    logger.info("开始后处理")

    result=data_util(cls_result,ner_ctc_texts,single_true_result_ids,cged_test_data,ner_cls_result,single_texts)
    logger.info("后处理完毕")

    result=[i+'\n' for i in result]

    with open('cged_pred.txt','w',encoding='utf-8') as f:
        f.writelines(result)

    logger.info("任务完成")


if __name__=='__main__':
    main()
