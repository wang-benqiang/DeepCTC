#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import time
from collections import Iterable
from math import ceil, log
from typing import List

import numpy as np
import torch
from rich.progress import track
from sklearn.metrics import f1_score
import logger
from deeplearning.dataset.csc_bert_dataset import DatasetCscSeq2label
from deeplearning.modeling.modeling_ctc import ModelingCTC
from evaluate.metrics.metric_csc import f1_csc
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AdamW
from deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer


class TrainerCscBert:
    def __init__(self,
                 model_dir,
                 model_type,
                 dropout,
                 lr,
                 lr_change_ratio_strategy,
                 first_batch_warm_up_lr,
                 epoch_strategy_for_adjust_lr,
                 random_seed_num,
                 epochs,
                 batch_size,
                 max_seq_len,
                 pretrained_model_dir,
                 eval_ratio_of_epoch,
                 early_stop_times=20,
                 max_dataset_len=None,
                 enable_pretrain=True,
                 loss_ignore_index=-100,
                 ctc_label_vocab_dir='src/deeplearning/ctc_vocab',
                 freeze_embedding=True,
                 dev_data_ratio=None,
                 parallel_training=False,
                 amp=True,
                 nodes_num=1,
                 ddp_local_rank=-1,
                 with_train_epoch_metric=True,
                 cuda_id=None,
                 train_db_dirs=['src/deeplearning/dataset/lm_db/db/csc_train'],
                 dev_db_dirs=['src/deeplearning/dataset/lm_db/db/csc_dev'],
                 test_db_dirs=['src/deeplearning/dataset/lm_db/db/csc_test'],
                 balanced_loss=False,
                 use_cuda=False,
                 **kwargs
                 ):

        """[summary]

        Args:
            trainer_type ([type]): [训练器类型: (detector, corrector, dc, mlm)]
            model_dir ([type]): [模型保存目录]
            model_name ([type]): [模型名称]
            dropout ([type]): [description]
            lr ([type]): [初始学习率]
            lr_change_ratio_strategy ([type]): [学习率变化策略, 可以是一个浮点数或者一个浮点数列表, 当满足epoch条件后, 如果是整数,会将学习率按照浮点数固定变化, 如果是列表则会用next()迭代调用不同的浮点数进行变化, 当迭代完毕则不再变化]
            epoch_strategy_for_adjust_lr ([type]): [学习率变化对应的epoch条件, 可以是一个整数或者一个整数列表, 如果是整数, 表示每多少个epoch变化学习率, 如果是列表, 表示指定了哪些epoch来调整学习率]
            random_seed_num ([type]): [随机种子]]
            epochs ([type]): [训练周期数]]
            batch_size ([type]): [description]
            max_seq_len ([type]): [description]
            train_file_dir ([type]): [description]
            pretrained_model_dir ([type]): [预训练模型目录]
            eval_ratio_of_epoch ([type]): [每多少个epoch进行验证，可以是浮点数]]
            early_stop_times (int, optional): [description]. Defaults to 20.
            max_dataset_length ([type], optional): [最大数据集数量, 测试时候可以将数据集大小改为10000条进行测试]. Defaults to None.
            freeze_embedding (bool, optional): [description]. Defaults to True.
            dev_data_ratio ([type], optional): [description]. Defaults to None.
            parallel_training (bool, optional): [description]. Defaults to False.
            cuda_id ([type], optional): [description]. Defaults to None.
        """

        current_time = time.strftime("_%YY%mM%dD%HH", time.localtime())
        self.model_dir = os.path.join(model_dir, '')[:-1] + current_time
        self.loss_ignore_index = loss_ignore_index
        self.ddp_local_rank = int(ddp_local_rank)
        self.cuda_id = cuda_id
        self.use_cuda = use_cuda
        self.nodes_num = nodes_num
        self.balanced_loss = balanced_loss
        self.model_type = model_type
        if not os.path.exists(self.model_dir) and self.ddp_local_rank in (-1, 0):
            os.mkdir(self.model_dir)

        self.num_labels = None
        self.dropout = dropout
        self.first_batch_warm_up_lr = first_batch_warm_up_lr
        if self.first_batch_warm_up_lr is not None:
            self.has_warm_up_ed = False
        self.lr = lr
        self.lr_change_ratio_strategy = lr_change_ratio_strategy
        self.lr_change_ratio_strategy_iteration = iter(
            lr_change_ratio_strategy) if isinstance(lr_change_ratio_strategy,
                                                    Iterable) else None
        self.epoch_strategy_for_adjust_lr = epoch_strategy_for_adjust_lr
        self.epoch_strategy_for_adjust_lr_iteration = iter(
            epoch_strategy_for_adjust_lr) if isinstance(
            epoch_strategy_for_adjust_lr, Iterable) else None
        self.random_seed_num = random_seed_num
        self.epochs = epochs
        self.batch_size = batch_size
        self.with_train_epoch_metric = with_train_epoch_metric
        self.max_seq_len = max_seq_len
        self.max_dataset_len = max_dataset_len

        self.dev_data_ratio = dev_data_ratio
        self.enable_pretrain = enable_pretrain

        self.pretrained_model_dir = pretrained_model_dir
        self.eval_ratio_of_epoch = eval_ratio_of_epoch
        self._eval_steps = None  # 后面加载数据的时候会按照比例计算每多少step验证一次
        self.early_stop_times = early_stop_times
        self.freeze_embedding = freeze_embedding
        self.parallel_training = parallel_training
        self.amp = amp
        self.ctc_label_vocab_dir = ctc_label_vocab_dir

        self.train_db_dirs = train_db_dirs
        self.dev_db_dirs = dev_db_dirs
        self.test_db_dirs = test_db_dirs

        self._dev_size = None
        self._train_size = None
        if self.amp:
            self.scaler = GradScaler()  # auto mixed precision

        self.tokenizer = CustomBertTokenizer.from_pretrained(
            self.pretrained_model_dir)
        self.model = self.load_model()
        self.model_config = self.model.config
        # load_data 会用到modelconfig
        self.optimizer = self.load_suite()

        self._id2dtag, self._dtag2id, self._id2ctag, self._ctag2id = self.load_label_dict()

        self._keep_c_tag_id = self._ctag2id['$KEEP']

    def load_label_dict(self):
        dtag_fp = os.path.join(self.ctc_label_vocab_dir, 'ctc_detect_2tags.txt')
        ctag_fp = os.path.join(self.ctc_label_vocab_dir, 'ctc_correct_tags.txt')

        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        # logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id

    def load_data(self, ith_train_db) -> List[DataLoader]:

        # RealiseDataset

        train_ds = DatasetCscSeq2label(
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            max_dataset_len=self.max_dataset_len,
            db_dir=self.train_db_dirs[ith_train_db],
            balanced_loss=self.balanced_loss,
            enable_pretrain=self.enable_pretrain,
        )

        self.num_labels = self.tokenizer.vocab_size
        if self.dev_db_dirs is None or not os.path.exists(self.dev_db_dirs[0]):

            # 如果没有dev集,则会切分训练集
            self._dev_size = int(len(train_ds) * self.dev_data_ratio)
            self._train_size = len(train_ds) - self._dev_size

            train_ds, dev_ds = torch.utils.data.random_split(
                train_ds, [self._train_size, self._dev_size])
        else:
            dev_ds = DatasetCscSeq2label(
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                max_dataset_len=self.max_dataset_len,
                balanced_loss=False,
                enable_pretrain=self.enable_pretrain,
            )
        if os.path.exists(self.test_db_dirs[0]):
            test_ds = DatasetCscSeq2label(
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                max_dataset_len=self.max_dataset_len,
                db_dir=self.test_db_dirs[0],
                balanced_loss=False,
                enable_pretrain=self.enable_pretrain,
            )
        else:
            test_ds = None

        self._dev_size = len(dev_ds)
        self._train_size = len(train_ds)
        if test_ds is not None:
            self._test_size = len(test_ds)

        self._train_steps = ceil(self._train_size / self.batch_size)
        self._eval_steps = ceil(self.eval_ratio_of_epoch * self._train_steps)

        self._train_steps = ceil(self._train_steps / self.nodes_num)
        self._eval_steps = ceil(self._eval_steps / self.nodes_num)

        # if self._eval_steps < 10:
        #     self._eval_steps = 10
        # logger.info('cuda nodes num:{}'.format(self.nodes_num))
        # logger.info('_train_size:{}'.format(self._train_size))
        # logger.info('_dev_size:{}'.format(self._dev_size))
        # if test_ds is not None:
        #     logger.info('_test_size:{}'.format(self._test_size))
        # logger.info('Steps of one epoch : {}'.format(self._train_steps))
        # logger.info('Evaluation every {} steps'.format(self._eval_steps))

        if self.ddp_local_rank != -1:
            train_ds = torch.utils.data.dataloader.DataLoader(
                train_ds, sampler=DistributedSampler(train_ds), batch_size=self.batch_size, num_workers=8,
                prefetch_factor=2)

            dev_ds = torch.utils.data.dataloader.DataLoader(
                dev_ds, batch_size=self.batch_size, shuffle=False, num_workers=8, prefetch_factor=2)

            test_ds = torch.utils.data.dataloader.DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=False, num_workers=8, prefetch_factor=2)
        else:
            train_ds = torch.utils.data.dataloader.DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8, prefetch_factor=2)
            dev_ds = torch.utils.data.dataloader.DataLoader(
                dev_ds, batch_size=self.batch_size, shuffle=False, num_workers=8, prefetch_factor=2)
            test_ds = torch.utils.data.dataloader.DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=False, num_workers=8, prefetch_factor=2)

        return [train_ds, dev_ds, test_ds]

    def load_suite(self):
        "需要先加载模型"
        model_params = list(self.model.named_parameters())
        if self.freeze_embedding:
            # logger.info('freeze_embedding!')
            model_params = filter(lambda p: p[1].requires_grad, model_params)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in model_params
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                0.01
        }, {
            'params':
                [p for n, p in model_params if any(nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        return optimizer

    def load_model(self) -> ModelingCTC:
        # Realise
        model = ModelingCTC.from_pretrained(
            self.pretrained_model_dir)
        if self.freeze_embedding:
            embedding_name_list = ('embeddings.word_embeddings.weight',
                                   'embeddings.position_embeddings.weight',
                                   'embeddings.token_type_embeddings.weight')
            for named_para in model.named_parameters():
                named_para[1].requires_grad = False if named_para[
                                                           0] in embedding_name_list else True
        logger.info('model loaded from {}'.format(self.pretrained_model_dir))
        return model

    def save_model(self, saved_dir):
        model_to_save = self.model.module if hasattr(
            self.model, 'module') else self.model
        if self.ddp_local_rank in (-1, 0):
            if not os.path.exists(saved_dir):
                os.mkdir(saved_dir)
            model_to_save.save_pretrained(saved_dir)
            self.tokenizer.save_pretrained(saved_dir)
            logger.info('=========== Best Model saved at {} ============='.format(
                saved_dir))

    def get_config(self):
        """

        """
        config = {
            "model_dir": self.model_dir,
            "lr": self.lr,
            "lr_change_ratio_strategy": self.lr_change_ratio_strategy,
            "first_batch_warm_up_lr": self.first_batch_warm_up_lr,
            'loss_ignore_index': self.loss_ignore_index,  # loss忽略-1
            "epoch_strategy_for_adjust_lr": self.epoch_strategy_for_adjust_lr,
            "random_seed_num": self.random_seed_num,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "eval_ratio_of_epoch": self.eval_ratio_of_epoch,
            "early_stop_times": self.early_stop_times,
            "max_seq_len": self.max_seq_len,
            "freeze_embedding": self.freeze_embedding,
            "_train_size": self._train_size,
            "eval_ratio_of_epoch": self.eval_ratio_of_epoch,
            "pretrained_model_dir": self.pretrained_model_dir,
            "parallel_training": self.parallel_training,
            "train_db_dirs": self.train_db_dirs,
            "test_db_dirs": self.test_db_dirs,
        }
        return config

    def save_config(self):
        if self.ddp_local_rank in (-1, 0):
            fp = '{}/train_config.json'.format(self.model_dir)
            with open(fp, 'w', encoding='utf8') as f:
                json.dump(self.get_config(), f, indent=4)
            logger.info('train config has been saved at {}'.format(fp))

    @staticmethod
    def fit_seed(random_seed_num):
        np.random.seed(random_seed_num)
        torch.manual_seed(random_seed_num)
        torch.cuda.manual_seed_all(random_seed_num)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    def train(self):
        """[summary]

        Args:
            wait_cuda_memory (bool, optional): []. Defaults to False.

        Returns:
            [type]: [description]
        """
        self.save_config()
        self.fit_seed(self.random_seed_num)
        self.equip_cuda()
        best_eval_score = 0
        early_stop_time = 0
        final_eval_scores_for_early_stop = []

        epoch_end_flag = False  # 每个epoch再验证
        for epoch in range(1, self.epochs + 1):

            if self.with_train_epoch_metric:
                epoch_preds, epoch_gold_labels = [], []
            else:
                epoch_preds, epoch_gold_labels = None, None
            epoch_c_loss = 0

            for ith_train_db in range(len(self.train_db_dirs)):
                logger.info("Epoch {}, start to load {}th train db.".format(
                    epoch, ith_train_db))
                self.train_ds, self.dev_ds, self.test_ds = self.load_data(
                    ith_train_db)
                steps = ceil(self._train_size / self.batch_size)
                if self.ddp_local_rank != -1:
                    self.train_ds.sampler.set_epoch(epoch)

                for step, batch_ds in track(enumerate(self.train_ds),
                                            description='Training',
                                            total=self._train_steps):
                    step += 1
                    # 如果显存不够, 开始循环等待显存够了再训
                    trained_flag = False
                    while not trained_flag:
                        try:

                            batch_loss, batch_gold, batch_pred = self.train_step(
                                batch_ds, return_for_epoch_metric=self.with_train_epoch_metric)
                            trained_flag = True
                        except RuntimeError as e:
                            logger.exception(e)
                            logger.error(
                                'error continue')
                            continue
                            # self.clear_cuda()  # 先释放cuda给他人使用
                            # time.sleep(60 * 30)
                            # self.equip_cuda()
                    if self.with_train_epoch_metric:
                        epoch_preds += batch_pred
                        epoch_gold_labels += batch_gold
                    epoch_c_loss += batch_loss
                    if (step % self._eval_steps == 0 or epoch_end_flag) and self.ddp_local_rank in (-1, 0):
                        # TODO
                        logger.info('[Start Evaluating]: local rank {}'.format(
                            self.ddp_local_rank))
                        epoch_end_flag = False
                        eval_epoch_c_loss, eval_epoch_precision, eval_epoch_recall, eval_epoch_f1 = self.evaluate(
                            data_type='dev')
                        test_epoch_c_loss, test_epoch_precision, test_epoch_recall, test_epoch_f1 = self.evaluate(
                            data_type='test')
                        log_text = '[Evaluating] Epoch {}/{}, Step {}/{}, ' \
                                   'epoch_c_loss:{}, epoch_precision:{}, epoch_recall:{}, epoch_f1:{},  '
                        logger.info(
                            log_text.format(epoch, self.epochs, step, steps,
                                            eval_epoch_c_loss, eval_epoch_precision,
                                            eval_epoch_recall, eval_epoch_f1,
                                            ))
                        log_text = '[Testing] Epoch {}/{}, Step {}/{}, ' \
                                   'epoch_c_loss:{}, epoch_precision:{}, epoch_recall:{}, epoch_f1:{}'
                        logger.info(
                            log_text.format(epoch, self.epochs, step, steps,
                                            test_epoch_c_loss, test_epoch_precision,
                                            test_epoch_recall, test_epoch_f1,
                                            ))

                        if eval_epoch_f1 >= 0:
                            best_eval_score = eval_epoch_f1
                            test_f1_str = str(round(test_epoch_f1 * 100,
                                                    2)).replace('.', '_') + '%'
                            dev_f1_str = str(round(eval_epoch_f1 * 100,
                                                   2)).replace('.', '_') + '%'
                            metric_str = 'epoch{},ith_db{},step{},testf1_{},devf1_{}'.format(epoch, ith_train_db, step,
                                                                                             test_f1_str, dev_f1_str)
                            saved_dir = os.path.join(
                                self.model_dir, metric_str)
                            self.save_model(saved_dir)
                            early_stop_time = 0
                            final_eval_scores_for_early_stop = []
                            if eval_epoch_f1 >= 1:
                                logger.info(
                                    'Eval epoch f1-score has reached to 1.0, continue training')
                                pass
                        else:
                            early_stop_time += 1
                            final_eval_scores_for_early_stop.append(
                                eval_epoch_f1)
                            if early_stop_time >= self.early_stop_times:
                                logger.info(
                                    '[Early Stop], final eval_score:{}'.format(
                                        final_eval_scores_for_early_stop))
                                return 1
            if self.with_train_epoch_metric:
                "因为输出和输入空间不一样，所以计算指标要对应输出空间，原字符输出对应keep"
                epoch_src = [self._id2ctag[self._keep_c_tag_id]] * len(epoch_src)
                (d_precision, d_recall, d_f1), (c_precision, c_recall, c_f1) = f1_csc(
                    src_texts=[epoch_src], trg_texts=[epoch_gold_labels], pred_texts=[epoch_preds])

            else:
                epoch_precision, epoch_recall, epoch_f1 = None, None, None

            if self.ddp_local_rank in (-1, 0):
                logger.info('Epoch End..')
                epoch_end_flag = True
                log_text = '[Training epoch] Epoch {}/{},' \
                           'epoch_c_loss:{}, epoch_precision:{}, epoch_recall:{}, epoch_f1:{}'
                logger.info(
                    log_text.format(epoch, self.epochs, epoch_c_loss,
                                    epoch_precision, epoch_recall, epoch_f1))
            # 调整学习率
            self.adjust_lr(epoch)
            # r = classification_report(epoch_gold_labels, epoch_preds, target_names=['CORRECT', 'INCORRECT'])

        return 1

    def equip_cuda(self):
        try:
            if torch.cuda.is_available() and self.use_cuda:
                if self.cuda_id is not None:
                    torch.cuda.set_device(self.cuda_id)
                self.model.cuda()
                # self.criterion.cuda()
                device_count = torch.cuda.device_count()
                devices_ids = list(range(device_count))
                if self.parallel_training and device_count > 1:
                    self.model = torch.nn.DataParallel(self.model,
                                                       device_ids=devices_ids)
                    logger.info('DP training, use cuda list:{}'.format(
                        devices_ids))
                elif self.ddp_local_rank != -1:
                    self.model = DDP(self.model, device_ids=[int(
                        self.ddp_local_rank)], output_device=int(self.ddp_local_rank), find_unused_parameters=True)
                    logger.info('DDP training, use cuda list:{}'.format(
                        devices_ids))
                else:
                    logger.info('Use single cuda for training')
            else:
                logger.info('use cpu to train')
        except RuntimeError as e:
            logger.exception(e)
            logger.warn('CUDA out of memery, start to wait for 30min')
            self.clear_cuda()  # 先释放cuda给他人使用
            time.sleep(60 * 30)
            self.equip_cuda()

    def clear_cuda(self):
        self.model.cpu()
        # self.criterion.cpu()
        logger.info('CUDA Memory has been cleared..')

    def adjust_lr(self, current_epoch):
        "根据epoch策略和学习率策略调整学习率"

        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info('[Learning rate Ajuster start]')
        logger.info('current_epoch:{}, epoch_strage:{}'.format(
            current_epoch, self.epoch_strategy_for_adjust_lr))

        "先判断epoch是否达到条件"
        if self.epoch_strategy_for_adjust_lr_iteration is None:
            # 每多少个epoch进行调整
            if current_epoch % self.epoch_strategy_for_adjust_lr != 0:
                logger.info('epoch does not matched!')
                logger.info('[LR]: keep with {} in'.format(current_lr))
                return 0
        else:
            # 指定某些epoch调整
            try:
                trg_epoch = next(self.epoch_strategy_for_adjust_lr_iteration)
                if current_epoch != trg_epoch:
                    logger.info('epoch does not matched')
                    logger.info('[LR]: keep with {} in'.format(current_lr))
            except StopIteration:
                logger.info('epoch_strategy StopIteration: {}'.format(
                    self.epoch_strategy_for_adjust_lr))
                logger.info('[LR]: keep with {} in'.format(current_lr))
                return 0

        logger.info('The epoch meet the condition to adjust lr')
        logger.info('initial lr: {}'.format(self.lr))
        logger.info('lr_change_ratio_strategy: {}'.format(
            self.lr_change_ratio_strategy))

        "epoch达到改变学习率的条件以后开始调整学习率"
        if self.lr_change_ratio_strategy_iteration is None:
            # 学习率按照同一个比例变化
            trg_lr = current_lr * self.lr_change_ratio_strategy
        else:
            # 学习率按照各自的比例变化
            try:
                trg_lr = current_lr * next(
                    self.lr_change_ratio_strategy_iteration)
            except StopIteration:
                logger.info('lr_change_ratio_strategy StopIteration')
                logger.info('[LR]: keep with {} '.format(current_lr))
                return 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = trg_lr
        logger.info('[LR changed]: from {} to {}'.format(current_lr, trg_lr))
        return 1

    def warm_up(self, action='start'):
        assert action in ('start', 'stop')
        if action == 'start' and self.first_batch_warm_up_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.first_batch_warm_up_lr

            logger.info('[Warm Up Start]: lr start at {}'.format(
                self.first_batch_warm_up_lr))
            return 1
        elif action == 'stop':
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            logger.info('[Warm Up End]: lr restore to {}'.format(self.lr))
            return 1

    def train_step(self, batch_ds, return_for_epoch_metric=True):

        self.model.train()

        if self.first_batch_warm_up_lr is not None:
            if self.has_warm_up_ed == False:
                self.warm_up(action='start')

        if torch.cuda.is_available() and self.use_cuda:
            batch_ds['input_ids'] = batch_ds['input_ids'].cuda()
            batch_ds['attention_mask'] = batch_ds['attention_mask'].cuda()
            batch_ds['token_type_ids'] = batch_ds['token_type_ids'].cuda()
            batch_ds['c_tags'] = batch_ds['c_tags'].cuda()
            batch_ds['d_tags'] = batch_ds['d_tags'].cuda()

        self.optimizer.zero_grad()
        c_logits, d_logits, batch_loss = self.model(
            input_ids=batch_ds['input_ids'],
            attention_mask=batch_ds['attention_mask'],
            token_type_ids=batch_ds['token_type_ids'],
            detect_labels=batch_ds['d_tags'],
            correct_labels=batch_ds['c_tags'],
        )
        batch_loss = batch_loss.mean()
        if self.parallel_training and torch.cuda.is_available(
        ) and torch.cuda.device_count() > 1:
            try:
                batch_loss = torch.sum(batch_loss) / len(batch_loss)
            except:
                pass
        if self.amp:
            "修改为混合精度"
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            batch_loss.backward()
            self.optimizer.step()
        if self.first_batch_warm_up_lr is not None:
            if self.has_warm_up_ed == False:
                self.warm_up(action='stop')
                self.has_warm_up_ed = True
        if return_for_epoch_metric:
            batch_gold = batch_ds['c_tags'].view(-1).tolist()
            batch_pred = torch.argmax(c_logits,
                                      dim=-1).view(-1).tolist()

            seq_true_idx = np.argwhere(batch_gold != self.loss_ignore_index)
            batch_gold = batch_gold[seq_true_idx].squeeze()
            batch_pred = batch_pred[seq_true_idx].squeeze()

            return batch_loss.item(), list(batch_gold), list(batch_pred)
        else:

            return batch_loss.item(), None, None

    @torch.no_grad()
    def evaluate(self, data_type='dev'):
        # 分布式训练时, 外层调用前会确认节点为-1,0时, 才会做验证
        self.model.eval()
        epoch_c_loss = 0
        epoch_preds, epoch_gold_labels, epoch_src = [], [], []

        detect_epoch_preds, detect_epoch_gold_labels = [], []

        if data_type == 'dev':
            dataset = self.dev_ds
        else:
            dataset = self.test_ds

        for batch_ds in dataset:
            if torch.cuda.is_available() and self.use_cuda:
                batch_ds['input_ids'] = batch_ds['input_ids'].cuda()
                batch_ds['attention_mask'] = batch_ds['attention_mask'].cuda()
                batch_ds['c_tags'] = batch_ds['c_tags'].cuda()
                batch_ds['token_type_ids'] = batch_ds['token_type_ids'].cuda()

                batch_ds['d_tags'] = batch_ds['d_tags'].cuda()

            c_logits, d_logits, batch_loss = self.model(
                input_ids=batch_ds['input_ids'],
                attention_mask=batch_ds['attention_mask'],
                token_type_ids=batch_ds['token_type_ids'],
                detect_labels=batch_ds['d_tags'],
                correct_labels=batch_ds['c_tags'],
            )

            batch_loss = batch_loss.mean()

            # correct

            batch_gold = batch_ds['c_tags'].view(-1).cpu().numpy()
            batch_pred = torch.argmax(c_logits,
                                      dim=-1).view(-1).cpu().numpy()
            batch_src = batch_ds['input_ids'].view(-1).cpu().numpy()

            seq_true_idx = np.argwhere(batch_gold != self.loss_ignore_index)  # 获取非pad部分的标签

            batch_gold = batch_gold[seq_true_idx].squeeze()
            batch_pred = batch_pred[seq_true_idx].squeeze()
            batch_src = batch_src[seq_true_idx].squeeze()

            epoch_src += list(batch_src)

            epoch_gold_labels += list(batch_gold)
            epoch_preds += list(batch_pred)

            # detect
            if self.model_config.add_detect_task:
                detect_batch_gold = batch_ds['d_tags'].view(-1).cpu().numpy()
                detect_batch_pred = torch.argmax(d_logits,
                                                 dim=-1).view(-1).cpu().numpy()
                detect_batch_gold = detect_batch_gold[seq_true_idx].squeeze()
                detect_batch_pred = detect_batch_pred[seq_true_idx].squeeze()

                detect_epoch_gold_labels += list(detect_batch_gold)
                detect_epoch_preds += list(detect_batch_pred)

            epoch_c_loss += batch_loss.item()

            "因为输出和输入空间不一样，所以计算指标要对应输出空间，原字符输出对应keep"
            epoch_src = [self._keep_c_tag_id] * len(epoch_src)
        (d_precision, d_recall, d_f1), (c_precision, c_recall, c_f1) = f1_csc(
            src_texts=[epoch_src], trg_texts=[epoch_gold_labels], pred_texts=[epoch_preds])

        detect_f1 = f1_score(detect_epoch_gold_labels,
                             detect_epoch_preds,
                             labels=[1],
                             average='macro',
                             zero_division=0)

        logger.info('[{}, detect task]: nokeep f1 score:{}%'.format(
            data_type, detect_f1 * 100))

        return epoch_c_loss, c_precision, c_recall, c_f1
