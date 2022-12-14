#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import time
from collections import Iterable
from math import ceil
from typing import List

import numpy as np
import torch
from rich.progress import track
from sklearn.metrics import f1_score, precision_score, recall_score
from src import logger
from src.deeplearning.dataset.seq2label_dataset import Seq2labelDataset
from src.deeplearning.loss.focal_loss import FocalCELoss
from src.deeplearning.modeling.modeling_seq2label import Sequence2Label
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers.models.bert import BertTokenizer
from transformers.models.electra import ElectraTokenizer
from torch.cuda.amp import GradScaler


class TrainerSeq2label:
    def __init__(self,
                 model_dir,
                 model_name,
                 dropout,
                 lr,
                 lr_change_ratio_strategy,
                 first_batch_warm_up_lr,
                 epoch_strategy_for_adjust_lr,
                 random_seed_num,
                 epochs,
                 batch_size,
                 max_seq_len,
                 keep_one_append,
                 train_file_dir,
                 dev_file_dir,
                 test_file_dir,
                 ctc_label_vocab_dir,
                 pretrained_model_type,
                 pretrained_model_dir,
                 eval_ratio_of_epoch,
                 only_load_pretrained_config=False,
                 two_cls_layer=False,
                 early_stop_times=20,
                 d_tag_type='all',
                 max_dataset_length=None,
                 res_mode=True,
                 loss_labels_weights=None,
                 focal_loss_gamma=2,
                 loss_ignore_index=-1,
                 freeze_embedding=True,
                 dev_data_ratio=None,
                 parallel_training=False,
                 cuda_id=None,
                 train_ds_pkl='train_ds.pkl',
                 dev_ds_pkl='dev_ds.pkl',
                 test_ds_pkl='test_ds.pkl',
                  ):
        """[summary]

        Args:
            trainer_type ([type]): [???????????????: (detector, corrector, dc, mlm)]
            model_dir ([type]): [??????????????????]
            model_name ([type]): [????????????]
            dropout ([type]): [description]
            lr ([type]): [???????????????]
            lr_change_ratio_strategy ([type]): [?????????????????????, ???????????????????????????????????????????????????, ?????????epoch?????????, ???????????????,??????????????????????????????????????????, ????????????????????????next()??????????????????????????????????????????, ??????????????????????????????]
            epoch_strategy_for_adjust_lr ([type]): [????????????????????????epoch??????, ?????????????????????????????????????????????, ???????????????, ??????????????????epoch???????????????, ???????????????, ?????????????????????epoch??????????????????]
            random_seed_num ([type]): [????????????]]
            epochs ([type]): [???????????????]]
            batch_size ([type]): [description]
            max_seq_len ([type]): [description]
            keep_one_append ([type]): [????????????append, ?????????????????????append??????????????????]
            train_file_dir ([type]): [description]
            dev_file_dir ([type]): [description]
            ctc_label_vocab_dir ([type]): [description]
            pretrained_model_type ([type]): [?????????????????????]
            pretrained_model_dir ([type]): [?????????????????????]
            eval_ratio_of_epoch ([type]): [????????????epoch?????????????????????????????????]]
            early_stop_times (int, optional): [description]. Defaults to 20.
            max_dataset_length ([type], optional): [?????????????????????, ??????????????????????????????????????????10000???????????????]. Defaults to None.
            res_mode (bool, optional): [description]. Defaults to True.
            freeze_embedding (bool, optional): [description]. Defaults to True.
            dev_data_ratio ([type], optional): [description]. Defaults to None.
            parallel_training (bool, optional): [description]. Defaults to False.
            cuda_id ([type], optional): [description]. Defaults to None.
       
        """

        current_time = time.strftime("_%Y-%m-%d_%Hh", time.localtime())
        d_tag_types = ('all', 'replace', 'redundant', 'miss')
        assert d_tag_type in d_tag_types, 'keep d_tag_type in {}'.format(
            d_tag_types)
        self.d_tag_type = d_tag_type
        self.model_dir = model_dir + current_time + '_created'
        self.loss_labels_weights = loss_labels_weights
        self.focal_loss_gamma = focal_loss_gamma
        self.loss_ignore_index = loss_ignore_index

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.model_name = model_name
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
        self.max_seq_len = max_seq_len
        self.max_dataset_length = max_dataset_length
        self.keep_one_append = keep_one_append
        self.train_file_dir = train_file_dir
        self.dev_file_dir = dev_file_dir  # dev????????????dev?????? ?????? ?????????????????????????????????
        self.test_file_dir = test_file_dir  # test????????????test??????
        if self.dev_file_dir is None and dev_data_ratio is not None:
            self.dev_data_ratio = dev_data_ratio
        else:
            self.dev_data_ratio = None

        self.ctc_label_vocab_dir = ctc_label_vocab_dir
        self.pretrained_model_dir = pretrained_model_dir
        self.pretrained_model_type = pretrained_model_type
        self.eval_ratio_of_epoch = eval_ratio_of_epoch
        self._eval_steps = None  # ?????????????????????????????????????????????????????????step????????????
        self.early_stop_times = early_stop_times
        self.res_mode = res_mode
        self.scaler = GradScaler()
        self.only_load_pretrained_config = only_load_pretrained_config
        self.two_cls_layer = two_cls_layer
        self.freeze_embedding = freeze_embedding
        self.parallel_training = parallel_training
        self.cuda_id = cuda_id
        self.train_ds_pkl = train_ds_pkl
        self.dev_ds_pkl = dev_ds_pkl
        self.test_ds_pkl = test_ds_pkl
   
        self._dev_size = None
        self._train_size = None

        self.dtag_num = None

        pretrained_model_types = (
            'bert',
            'electra',
        )
        assert pretrained_model_type in pretrained_model_types, "pretrain_model_type should in {}".format(
            str(pretrained_model_types))
        if self.pretrained_model_type == 'electra':
            self.tokenizer = ElectraTokenizer.from_pretrained(
                self.pretrained_model_dir)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                self.pretrained_model_dir)
        self.train_ds, self.dev_ds, self.test_ds = self.load_data()
        self.model = self.load_model()

        self.criterion, self.optimizer = self.load_suite()

    def load_data(self) -> List[DataLoader]:

        train_ds_pkl_fp = '{}/{}'.format(
            self.model_dir,
            self.train_ds_pkl) if self.train_ds_pkl is not None else None
        dev_ds_pkl_fp = '{}/{}'.format(
            self.model_dir,
            self.dev_ds_pkl) if self.train_ds_pkl is not None else None

        test_ds_pkl_fp = '{}/{}'.format(
            self.model_dir,
            self.test_ds_pkl) if self.test_ds_pkl is not None else None

        if train_ds_pkl_fp is not None and os.path.exists(train_ds_pkl_fp):
            # ?????????????????????pkl,??????????????????pkl
            train_ds = Seq2labelDataset.load_ds_pkl(train_ds_pkl_fp)
        else:
            train_ds = Seq2labelDataset(
                self.train_file_dir,
                tokenizer=self.tokenizer,
                ctc_label_vocab_dir=self.ctc_label_vocab_dir,
                max_seq_len=self.max_seq_len,
                d_tag_type=self.d_tag_type,
                keep_one_append=self.keep_one_append,
                max_dataset_len=self.max_dataset_length,
                ds_pkl_fp=train_ds_pkl_fp)

        self.num_labels = train_ds.dtag_num

        if self.dev_file_dir is None:
            # ????????????dev???,?????????????????????
            self._dev_size = int(len(train_ds) * self.dev_data_ratio)
            self._train_size = len(train_ds) - self._dev_size
            train_ds, dev_ds = torch.utils.data.random_split(
                train_ds, [self._train_size, self._dev_size])
        else:
            # ?????????dev_file_dir???????????????dev pkl
            if dev_ds_pkl_fp is not None and os.path.exists(dev_ds_pkl_fp):
                # ?????????????????????pkl,??????????????????pkl
                dev_ds = Seq2labelDataset.load_ds_pkl(dev_ds_pkl_fp)
            else:
                dev_ds = Seq2labelDataset(
                    self.dev_file_dir,
                    tokenizer=self.tokenizer,
                    ctc_label_vocab_dir=self.ctc_label_vocab_dir,
                    max_seq_len=self.max_seq_len,
                    d_tag_type=self.d_tag_type,
                    keep_one_append=self.keep_one_append,
                    max_dataset_len=None,
                    ds_pkl_fp=dev_ds_pkl_fp)

        if self.test_file_dir is not None:

            if test_ds_pkl_fp is not None and os.path.exists(
                    test_ds_pkl_fp):
                # ?????????????????????pkl,??????????????????pkl
                test_ds = Seq2labelDataset.load_ds_pkl(test_ds_pkl_fp)
            else:
                test_ds = Seq2labelDataset(
                    self.test_file_dir,
                    tokenizer=self.tokenizer,
                    ctc_label_vocab_dir=self.ctc_label_vocab_dir,
                    max_seq_len=self.max_seq_len,
                    d_tag_type=self.d_tag_type,
                    keep_one_append=self.keep_one_append,
                    max_dataset_len=None,
                    ds_pkl_fp=test_ds_pkl_fp)
        else:
            test_ds = None

        self._dev_size = len(dev_ds)
        self._train_size = len(train_ds)
        if test_ds is not None:
            self._test_size = len(test_ds)

        self._train_steps = ceil(self._train_size / self.batch_size)
        self._eval_steps = ceil(self.eval_ratio_of_epoch * self._train_steps)
        if self._eval_steps < 100:
            self._eval_steps = 100
        logger.info('_train_size:{}'.format(self._train_size))
        logger.info('_dev_size:{}'.format(self._dev_size))
        if test_ds is not None:
            logger.info('_test_size:{}'.format(self._test_size))
        logger.info('Steps of one epoch : {}'.format(self._train_steps))
        logger.info('Evaluation every {} steps'.format(self._eval_steps))
        train_ds = torch.utils.data.dataloader.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True)
        dev_ds = torch.utils.data.dataloader.DataLoader(
            dev_ds, batch_size=self.batch_size, shuffle=True)
        test_ds = torch.utils.data.dataloader.DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False)
        return [train_ds, dev_ds, test_ds]

    def load_suite(self):
        "?????????????????????"
        criterion = FocalCELoss(loss_labels_weights=self.loss_labels_weights,
                                gamma=self.focal_loss_gamma,
                                ignore_index=self.loss_ignore_index)
        model_params = list(self.model.named_parameters())
        if self.freeze_embedding:
            logger.info('freeze_embedding!')
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
        return criterion, optimizer

    def load_model(self) -> Sequence2Label:
        model = Sequence2Label(
            pretrain_model_type=self.pretrained_model_type,
            pretrained_model_dir=self.pretrained_model_dir,
            num_labels=self.num_labels,
            dropout=self.dropout,
            res_mode=self.res_mode,
            only_load_pretrained_config=self.only_load_pretrained_config,
            two_cls_layer=self.two_cls_layer)
       
        if self.freeze_embedding:
            embedding_name_list = ('embeddings.word_embeddings.weight',
                                   'embeddings.position_embeddings.weight',
                                   'embeddings.token_type_embeddings.weight')
            for named_para in model.named_parameters():
                named_para[1].requires_grad = False if named_para[
                    0] in embedding_name_list else True
        return model

    def save_model(self, saved_fp):

        if self.parallel_training and torch.cuda.is_available(
        ) and torch.cuda.device_count() > 1:

            torch.save(self.model.module.state_dict(), saved_fp)
        else:
            torch.save(self.model.state_dict(), saved_fp)
        logger.info('=========== Best Model saved at {} ============='.format(
            saved_fp))

    def get_config(self):

        config = {
            "d_tag_type": self.d_tag_type,
            "model_dir": self.model_dir,
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "dtag_num": self.dtag_num,
            "lr": self.lr,
            "lr_change_ratio_strategy": self.lr_change_ratio_strategy,
            "first_batch_warm_up_lr": self.first_batch_warm_up_lr,
            'loss_labels_weights': self.loss_labels_weights,  # ??????label???loss??????,
            'focal_loss_gamma': self.focal_loss_gamma,  # focal_loss_gamma
            'loss_ignore_index': self.loss_ignore_index,  # loss??????-1
            "epoch_strategy_for_adjust_lr": self.epoch_strategy_for_adjust_lr,
            "random_seed_num": self.random_seed_num,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "eval_ratio_of_epoch": self.eval_ratio_of_epoch,
            "early_stop_times": self.early_stop_times,
            "res_mode": self.res_mode,
            "max_seq_len": self.max_seq_len,
            "freeze_embedding": self.freeze_embedding,
            "_train_size": self._train_size,
            "keep_one_append": self.keep_one_append,
            "train_file_dir": self.train_file_dir,
            "dev_file_dir": self.dev_file_dir,
            "test_file_dir": self.test_file_dir,
            "train_ds_pkl": self.train_ds_pkl,
            "dev_ds_pkl": self.dev_ds_pkl,
            "test_ds_pkl": self.test_ds_pkl,
            "dev_data_ratio": self.dev_data_ratio,
            "eval_ratio_of_epoch": self.eval_ratio_of_epoch,
            "ctc_label_vocab_dir": self.ctc_label_vocab_dir,
            "pretrained_model_type": self.pretrained_model_type,
            "pretrained_model_dir": self.pretrained_model_dir,
            "parallel_training": self.parallel_training,
            "train_db_dirs": self.train_db_dirs,
            "test_db_dirs": self.test_db_dirs,
        }
        return config

    def save_config(self):
        fp = '{}/train_config.json'.format(self.model_dir)
        with open(fp, 'w', encoding='utf8') as f:
            json.dump(self.get_config(), f, indent=4)
        logger.info('train config has been saved at {}'.format(fp))

    @staticmethod
    def fit_seed(random_seed_num):
        np.random.seed(random_seed_num)
        torch.manual_seed(random_seed_num)
        torch.cuda.manual_seed_all(random_seed_num)
        torch.backends.cudnn.deterministic = True  # ????????????????????????

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
        steps = ceil(self._train_size / self.batch_size)
        epoch_end_flag = False  # ??????epoch?????????
        for epoch in range(1, self.epochs + 1):
            epoch_preds, epoch_gold_labels = [], []
            epoch_loss = 0
            for step, batch_ds in track(enumerate(self.train_ds),
                                        description='Training',
                                        total=self._train_steps):
                step += 1
                # ??????????????????, ????????????????????????????????????
                trained_flag = False
                while not trained_flag:
                    try:
                        batch_loss, batch_gold, batch_pred = self.train_step(
                            batch_ds)

                        trained_flag = True
                    except RuntimeError as e:
                        logger.exception(e)
                        logger.warn(
                            'CUDA out of memery, start to wait for 30min')
                        self.clear_cuda()  # ?????????cuda???????????????
                        time.sleep(60 * 30)
                        self.equip_cuda()

                epoch_preds += batch_pred
                epoch_gold_labels += batch_gold
                epoch_loss += batch_loss

                if step % self._eval_steps == 0 or epoch_end_flag:
                    # TODO
                    epoch_end_flag = False
                    eval_epoch_loss, eval_epoch_precision, eval_epoch_recall, eval_epoch_f1, eval_epoch_f1_nokeep = self.evaluate(
                        data_type='dev')
                    test_epoch_loss, test_epoch_precision, test_epoch_recall, test_epoch_f1, test_epoch_f1_nokeep = self.evaluate(
                        data_type='test')
                    log_text = '[Evaluating] Epoch {}/{}, Step {}/{}, ' \
                               'epoch_loss:{}, epoch_precision:{}, epoch_recall:{}, epoch_f1:{}, epoch_f1_nokeep:{} '
                    logger.info(
                        log_text.format(epoch, self.epochs, step, steps,
                                        eval_epoch_loss, eval_epoch_precision,
                                        eval_epoch_recall, eval_epoch_f1,
                                        eval_epoch_f1_nokeep))
                    log_text = '[Testing] Epoch {}/{}, Step {}/{}, ' \
                               'epoch_loss:{}, epoch_precision:{}, epoch_recall:{}, epoch_f1:{}, epoch_f1_nokeep:{}'
                    logger.info(
                        log_text.format(epoch, self.epochs, step, steps,
                                        test_epoch_loss, test_epoch_precision,
                                        test_epoch_recall, test_epoch_f1,
                                        test_epoch_f1_nokeep))

                    if eval_epoch_f1 > best_eval_score:
                        best_eval_score = eval_epoch_f1

                        f1_str = str(round(test_epoch_f1_nokeep * 100,
                                           2)).replace('.', '_') + '%'
                        saved_fp = '{}/{}_testf1_{}.model'.format(
                            self.model_dir, self.model_name, f1_str)
                        self.save_model(saved_fp)
                        early_stop_time = 0
                        final_eval_scores_for_early_stop = []
                        if eval_epoch_f1 >= 1:
                            logger.info(
                                'Eval epoch f1-score has reached to 1.0 ')
                            return
                    else:
                        early_stop_time += 1
                        final_eval_scores_for_early_stop.append(eval_epoch_f1)
                        if early_stop_time >= self.early_stop_times:
                            logger.info(
                                '[Early Stop], final eval_score:{}'.format(
                                    final_eval_scores_for_early_stop))
                            return 1

            epoch_precision = precision_score(epoch_gold_labels,
                                              epoch_preds,
                                              average='macro',
                                              zero_division=0)
            epoch_recall = recall_score(epoch_gold_labels,
                                        epoch_preds,
                                        average='macro',
                                        zero_division=0)
            epoch_f1 = f1_score(epoch_gold_labels,
                                epoch_preds,
                                average=None,
                                zero_division=0)
            logger.info('Epoch End..')
            epoch_end_flag = True
            log_text = '[Training epoch] Epoch {}/{},' \
                       ' epoch_loss:{}, epoch_precision:{}, epoch_recall:{}, epoch_f1:{}'
            logger.info(
                log_text.format(epoch, self.epochs, epoch_loss,
                                epoch_precision, epoch_recall, epoch_f1))
            # ???????????????
            self.adjust_lr(epoch)
            # r = classification_report(epoch_gold_labels, epoch_preds, target_names=['CORRECT', 'INCORRECT'])

        return 1

    def equip_cuda(self):
        try:
            if torch.cuda.is_available():
                if self.cuda_id is not None:
                    torch.cuda.set_device(self.cuda_id)
                self.model.cuda()
                # self.criterion.cuda()
                device_count = torch.cuda.device_count()
                devices_ids = list(range(device_count))
                if self.parallel_training and device_count > 1:
                    self.model = torch.nn.DataParallel(self.model,
                                                       device_ids=devices_ids)
                    logger.info('Parallel training, use cuda list:{}'.format(
                        devices_ids))
                else:
                    logger.info('Use single cuda for training')
            else:
                logger.info('use cpu to train')
        except RuntimeError as e:
            logger.exception(e)
            logger.warn('CUDA out of memery, start to wait for 30min')
            self.clear_cuda()  # ?????????cuda???????????????
            time.sleep(60 * 30)
            self.equip_cuda()

    def clear_cuda(self):
        self.model.cpu()
        # self.criterion.cpu()
        logger.info('CUDA Memory has been cleared..')

    def adjust_lr(self, current_epoch):
        "??????epoch???????????????????????????????????????"

        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info('[Learning rate Ajuster start]')
        logger.info('current_epoch:{}, epoch_strage:{}'.format(
            current_epoch, self.epoch_strategy_for_adjust_lr))

        "?????????epoch??????????????????"
        if self.epoch_strategy_for_adjust_lr_iteration is None:
            # ????????????epoch????????????
            if current_epoch % self.epoch_strategy_for_adjust_lr != 0:
                logger.info('epoch does not matched!')
                logger.info('[LR]: keep with {} in'.format(current_lr))
                return 0
        else:
            # ????????????epoch??????
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

        "epoch?????????????????????????????????????????????????????????"
        if self.lr_change_ratio_strategy_iteration is None:
            # ????????????????????????????????????
            trg_lr = current_lr * self.lr_change_ratio_strategy
        else:
            # ????????????????????????????????????
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

    def train_step(self, batch_ds):
        self.model.train()

        if self.first_batch_warm_up_lr is not None:
            if self.has_warm_up_ed == False:
                self.warm_up(action='start')

        if torch.cuda.is_available():
            batch_ds['input_ids'] = batch_ds['input_ids'].cuda()
            batch_ds['attention_mask'] = batch_ds['attention_mask'].cuda()
            batch_ds['d_tags'] = batch_ds['d_tags'].cuda()
            batch_ds['token_type_ids'] = batch_ds['token_type_ids'].cuda()
        self.optimizer.zero_grad()

        batch_logits = self.model(
            input_ids=batch_ds['input_ids'],
            attention_mask=batch_ds['attention_mask'],
            token_type_ids=batch_ds['token_type_ids'],
        )

        batch_loss = self.criterion(batch_logits.view(-1, self.num_labels),
                                    batch_ds['d_tags'].view(-1))
                                    

        if self.parallel_training and torch.cuda.is_available(
        ) and torch.cuda.device_count() > 1:
            try:
                batch_loss = torch.sum(batch_loss) / len(batch_loss)
            except:
                pass

        "?????????????????????"
        self.scaler.scale(batch_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.first_batch_warm_up_lr is not None:
            if self.has_warm_up_ed == False:
                self.warm_up(action='stop')
                self.has_warm_up_ed = True
        batch_gold = batch_ds['d_tags'].view(-1).cpu().numpy()
        batch_pred = torch.argmax(batch_logits,
                                  dim=-1).view(-1).cpu().numpy()
        seq_true_idx = np.argwhere(batch_gold != -1)
        batch_gold = batch_gold[seq_true_idx].squeeze()
        batch_pred = batch_pred[seq_true_idx].squeeze()

        return batch_loss.item(), list(batch_gold), list(batch_pred)

    @torch.no_grad()
    def evaluate(self, data_type='dev'):
        # TODO
        self.model.eval()
        epoch_loss, epoch_precision, epoch_recall, epoch_f1 = 0, 0, 0, 0
        epoch_preds, epoch_gold_labels = [], []
        if data_type == 'dev':
            dataset = self.dev_ds
        else:
            dataset = self.test_ds

        for batch_ds in dataset:
            if torch.cuda.is_available():
                batch_ds['input_ids'] = batch_ds['input_ids'].cuda()
                batch_ds['attention_mask'] = batch_ds['attention_mask'].cuda()
                batch_ds['d_tags'] = batch_ds['d_tags'].cuda()
                batch_ds['token_type_ids'] = batch_ds[
                    'token_type_ids'].cuda()
            batch_logits = self.model(
                input_ids=batch_ds['input_ids'],
                attention_mask=batch_ds['attention_mask'],
                token_type_ids=batch_ds['token_type_ids'],
            )

            batch_loss = self.criterion(
                batch_logits.view(-1, self.num_labels),
                batch_ds['d_tags'].view(-1))

            batch_gold = batch_ds['d_tags'].view(-1).cpu().numpy()
            batch_pred = torch.argmax(batch_logits,
                                      dim=-1).view(-1).cpu().numpy()
            seq_true_idx = np.argwhere(batch_gold != -1)  # ?????????pad???????????????
            batch_gold = batch_gold[seq_true_idx].squeeze()
            batch_pred = batch_pred[seq_true_idx].squeeze()

            epoch_gold_labels += list(batch_gold)
            epoch_preds += list(batch_pred)

            if self.parallel_training and torch.cuda.is_available(
            ) and torch.cuda.device_count() > 1:
                try:
                    batch_loss = torch.sum(batch_loss) / len(batch_loss)
                except:
                    pass

            epoch_loss += batch_loss.item()

        epoch_precision = precision_score(epoch_gold_labels,
                                          epoch_preds,
                                          average='macro',
                                          zero_division=0)
        epoch_recall = recall_score(epoch_gold_labels,
                                    epoch_preds,
                                    average='macro',
                                    zero_division=0)
        epoch_f1 = f1_score(epoch_gold_labels,
                            epoch_preds,
                            average='macro',
                            zero_division=0)

        if self.d_tag_type == 'all':
            labels = [1, 2]
        elif self.d_tag_type in ['replace', 'miss', 'redundant']:
            labels = [1]
        epoch_f1_without_keep = f1_score(epoch_gold_labels,
                                         epoch_preds,
                                         labels=labels,
                                         average='macro',
                                         zero_division=0)

        return epoch_loss, epoch_precision, epoch_recall, epoch_f1, epoch_f1_without_keep
