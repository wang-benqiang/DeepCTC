
import time
import os
import json
import logging
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src.utils.trainer import train
from src.utils.options import Args
from src.utils.model_utils import build_model
from src.utils.dataset_utils import NERDataset
# from src.utils.evaluator import crf_evaluation, span_evaluation, mrc_evaluation
from src.utils.functions_utils import set_seed, get_model_path_list, load_model_and_parallel, get_time_dif
from src.preprocess.processor import convert_examples_to_features,read_txt
# from valid import valid_base

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def train_base(opt, train_examples, dev_examples=None):
    with open(os.path.join(opt.label_data_dir,'labels.txt'),'r', encoding='utf-8') as f:
        ent2id=f.readlines()
        ent2id = {j.strip():i for i,j in enumerate(ent2id)}

    with open(os.path.join(opt.label_data_dir,'d_tags.txt'),'r', encoding='utf-8') as f:
        dtag2id=f.readlines()
        dtag2id = {j.strip():i for i,j in enumerate(dtag2id)}


    train_features = convert_examples_to_features(train_examples,
                                                  opt.max_seq_len, opt.bert_dir, ent2id,dtag2id)[0]

    train_dataset = NERDataset(train_features, 'train', use_type_embed=opt.use_type_embed)


    model = build_model(opt.bert_dir, num_tags=len(ent2id),
                            num_dtags=len(dtag2id),
                            dropout_prob=opt.dropout_prob,
                            loss_type=opt.loss_type)

    train(opt, model, train_dataset)

def training(opt):
    train_examples = read_txt(opt.train_data_path)
    dev_examples = None
    if opt.eval_model:
        dev_examples = read_txt(opt.dev_data_path)

    bert_dir_list=opt.bert_dir_list.split(',')
    for bert_dir in bert_dir_list:
        opt.bert_dir=bert_dir
        train_base(opt, train_examples, dev_examples)



if __name__ == '__main__':
    start_time = time.time()
    logging.info('----------------开始计时----------------')
    logging.info('----------------------------------------')

    args = Args().get_parser()

    assert args.mode in ['train', 'stack'], 'mode mismatch'

    args.output_dir = os.path.join(args.output_dir, args.bert_type)

    set_seed(args.seed)

    if args.attack_train != '':
        args.output_dir += f'_{args.attack_train}'

    if args.weight_decay:
        args.output_dir += '_wd'

    if args.use_fp16:
        args.output_dir += '_fp16'


    args.output_dir += f'_{args.task_type}'

    # valid_base(args,read_txt(os.path.join(args.raw_data_dir, 'lang8_test.txt')))

    if args.mode == 'stack':
        args.output_dir += '_stack'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f'{args.mode} {args.task_type} in max_seq_len {args.max_seq_len}')

    training(args)


    time_dif = get_time_dif(start_time)
    logging.info("----------本次容器运行时长：{}-----------".format(time_dif))
