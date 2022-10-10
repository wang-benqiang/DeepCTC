#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import json
import os
import time
from math import ceil
from typing import List, Optional

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from logs import logger
from rich.progress import track
from src.dataset.dataset_ctc_seq2edit import DatasetCtcSeq2Edit
from src.metrics.metric_csc import f1_csc
from src.modeling.modeling_ctc_s2e_bert import ModelingCtcS2eBert
from src.tokenizer.bert_tokenizer import CustomBertTokenizer
from src.tricks.adversarial_training import FGM, PGD
from src.tricks.ema_torch import EMA
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


class TrainerCtcSeq2Edit:
    def __init__(self,
                 in_model_dir: str,
                 out_model_dir: str,
                 epochs: int,
                 batch_size: int,
                 learning_rate: float,
                 max_seq_len: int,
                 train_db_dir: str,
                 dev_db_dir: str = None,
                 test_db_dir: str = None,
                 max_dataset_len: int = -1,
                 random_seed_num: int = 42,
                 check_val_every_n_epoch: Optional[float] = 0.5,
                 early_stop_times: Optional[int] = 100,
                 freeze_embedding: bool = False,
                 freeze_backbone: bool = False,
                 warmup_steps: int = -1,
                 max_grad_norm: Optional[float] = None,
                 dev_data_ratio: Optional[float] = 0.2,
                 loss_ignore_id=-100,
                 mixed_precision='fp16',
                 ctc_label_vocab_dir: str = 'src/vocab',
                 gradient_accumulation_steps: Optional[int] = 1,
                 use_ema_train: bool = False,
                 use_rdrop_train: bool = False,
                 use_fgm_train: bool = False,
                 **kwargs
                 ):
        """
        # in_model_dir 预训练模型目录
        # out_model_dir 输出模型目录
        # epochs 训练轮数
        # batch_size batch文本数
        # max_seq_len 最大句子长度
        # learning_rate 学习率
        # train_fp 训练集文件
        # test_fp 测试集文件
        # dev_data_ratio  没有验证集时，会从训练集按照比例分割出验证集
        # random_seed_num 随机种子
        # warmup_steps 预热steps
        # check_val_every_n_epoch 每几轮对验证集进行指标计算
        # training_mode 训练模式 包括 ddp，dp, normal，分别代表分布式，并行，普通训练
        # mixed_precision choices=["no", "fp16", "bf16"],
        # freeze_embedding 是否冻结bert embed层
        """
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision, kwargs_handlers=[ddp_kwargs])
        current_time = time.strftime("_%YY%mM%dD%HH", time.localtime())
        self.in_model_dir = in_model_dir
        self.out_model_dir = os.path.join(out_model_dir, '')[
            :-1] + current_time

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_len = max_seq_len
        self.random_seed_num = random_seed_num
        self.freeze_embedding = freeze_embedding
        self.freeze_backbone = freeze_backbone
        self.train_db_dir = train_db_dir
        self.dev_db_dir = dev_db_dir
        self.test_db_dir = test_db_dir
        self.ctc_label_vocab_dir = ctc_label_vocab_dir
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.early_stop_times = early_stop_times
        self.dev_data_ratio = dev_data_ratio
        self._loss_ignore_id = loss_ignore_id
        self.max_dataset_len = max_dataset_len
        self._warmup_steps = warmup_steps
        self._max_grad_norm = max_grad_norm
        self._use_ema_train = use_ema_train
        self._use_rdrop_train = use_rdrop_train
        self._use_fgm_train = use_fgm_train
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.ema_model_path = os.path.join(self.out_model_dir, 'ema_model')
        self.normal_model_path = os.path.join(self.out_model_dir, 'normal_model')
        if not os.path.exists(self.out_model_dir) and self.accelerator.is_main_process:
            os.mkdir(self.out_model_dir)
            os.mkdir(self.ema_model_path)
            os.mkdir(self.normal_model_path)

        self.fit_seed(self.random_seed_num)
        self.tokenizer = CustomBertTokenizer.from_pretrained(self.in_model_dir)
        self.train_dataloader, self.dev_dataloader, self.test_dataloader = self.load_data()
        self.model, self.optimizer, self.lr_scheduler = self.load_suite()

        self._id2dtag, self._dtag2id, self._id2ctag, self._ctag2id = self.load_label_dict()
        self._keep_c_tag_id = self._ctag2id['$KEEP']
        self.equip_accelerator()
        if self._use_ema_train:
            self.ema_model = EMA(self.model, 0.9999, update_after_steps=10, update_every_steps=1)
            self.ema_model.register()
        if self._use_fgm_train:
            self.fgm = FGM(self.model)

        # save train params
        if self.accelerator.is_main_process:
            signature = inspect.signature(TrainerCtcSeq2Edit)
            train_kwargs = {}
            for param in signature.parameters.values():
                train_kwargs[param.name] = eval(param.name)
            train_kwargs = {k: v for k, v in train_kwargs.items(
            ) if isinstance(v, (str, bool, float, int))}

            out_fp = open(os.path.join(self.out_model_dir, 'train_config.json'),
                          'w', encoding='utf8')
            json.dump(train_kwargs, out_fp, ensure_ascii=False, indent=4)
            out_fp.close()

    def load_label_dict(self):
        dtag_fp = os.path.join(self.ctc_label_vocab_dir, 'ctc_detect_tags.txt')
        ctag_fp = os.path.join(self.ctc_label_vocab_dir,
                               'ctc_correct_tags.txt')

        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id

    def load_data(self) -> List[DataLoader]:

        # 加载train-dataset

        train_ds = DatasetCtcSeq2Edit(
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            db_dir=self.train_db_dir,
            max_dataset_len=self.max_dataset_len
        )

        if self.dev_db_dir is not None and os.path.exists(self.dev_db_dir):
            dev_ds = DatasetCtcSeq2Edit(
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                db_dir=self.dev_db_dir,
                max_dataset_len=self.max_dataset_len
            )
        else:
            # 如果没有dev set,则从训练集切分
            _dev_size = max(int(len(train_ds) * self.dev_data_ratio), 1)
            _train_size = len(train_ds) - _dev_size
            train_ds, dev_ds = torch.utils.data.random_split(
                train_ds, [_train_size, _dev_size])

        if self.test_db_dir is not None:
            test_ds = DatasetCtcSeq2Edit(
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                db_dir=self.test_db_dir,
                max_dataset_len=self.max_dataset_len
            )

        else:
            test_ds = None

        train_dataloader = torch.utils.data.dataloader.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8)
        dev_dataloader = torch.utils.data.dataloader.DataLoader(
            dev_ds, batch_size=self.batch_size, shuffle=False, num_workers=8)
        if test_ds is not None:
            test_dataloader = torch.utils.data.dataloader.DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=False, num_workers=8)

        return [train_dataloader, dev_dataloader, test_dataloader]

    def load_suite(self):
        "model"
        model = ModelingCtcS2eBert.from_pretrained(
            self.in_model_dir)
        model._init_criterion()
        if self.freeze_embedding:
            embedding_name_list = ('embeddings.word_embeddings.weight',
                                   'embeddings.position_embeddings.weight',
                                   'embeddings.token_type_embeddings.weight')
            for named_para in model.named_parameters():
                named_para[1].requires_grad = False if named_para[
                    0] in embedding_name_list else True

        if self.freeze_backbone:
            for named_para in model.named_parameters():
                named_para[1].requires_grad = False if '_cls' not in named_para[
                    0] else True
        "optimizer"
        # bert常用权重衰减
        model_params = list(model.named_parameters())
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

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        "scheduler"
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self._warmup_steps,
            num_training_steps=(len(self.train_dataloader) *
                                self.epochs) // self.gradient_accumulation_steps
        )
        return model, optimizer, lr_scheduler

    def save_model(self, out_model_dir):
        "保存模型"
        model_to_save = self.model.module if hasattr(
            self.model, 'module') else self.model
        if self.accelerator.is_main_process:
            if not os.path.exists(out_model_dir):

                os.mkdir(out_model_dir)
            model_to_save.save_pretrained(out_model_dir)
            self.tokenizer.save_pretrained(out_model_dir)
            logger.info('=========== New Model saved at {} ============='.format(
                out_model_dir))

    @staticmethod
    def fit_seed(random_seed_num):
        "固定随机种子 保证每次结果一样"
        np.random.seed(random_seed_num)
        torch.manual_seed(random_seed_num)
        torch.cuda.manual_seed_all(random_seed_num)
        torch.backends.cudnn.deterministic = True

    def train(self):
        """[summary]
        Args:
            wait_cuda_memory (bool, optional): []. Defaults to False.
        Returns:
            [type]: [description]
        """

        self._train_steps = len(self.train_dataloader)
        self._dev_steps = len(self.dev_dataloader)
        self._test_steps = len(
            self.test_dataloader) if self.test_dataloader is not None else 0

        logger.info('_train_steps:{}'.format(self._train_steps))
        logger.info('_dev_steps:{}'.format(self._dev_steps))
        logger.info('_test_steps:{}'.format(self._test_steps))
        # We need to keep track of how many total steps we have iterated over
        overall_step = 0
        # We also need to keep track of the stating epoch so files are named properly
        starting_epoch = 0

        max_eval_c_f1 = 0
        ith_early_stop_time = 0
        final_eval_scores_for_early_stop = []
        steps = self._train_steps
        epoch_end_flag = False  # 每个epoch再验证
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            for step, batch_ds in track(enumerate(self.train_dataloader),
                                        description='Training',
                                        total=len(self.train_dataloader)):
                step += 1
                overall_step += 1
                # if error, continue
                # try:
                batch_loss = self.train_step(
                    batch_ds, ith_step=step)
                # except RuntimeError as e:
                #     logger.error('ignore training step error!!')
                #     logger.exception(e)
                #     continue

                epoch_loss += batch_loss
                if (step % ceil(self.check_val_every_n_epoch * len(self.train_dataloader)) == 0 or epoch_end_flag):
                    #  到达验证步数，则开始验证，保存模型，记录最大的dev指标
                    logger.info('[Start Evaluating]')
                    epoch_end_flag = False
                    eval_epoch_loss, eval_epoch_c_f1 = self.evaluate(
                        dataset_type='dev', use_ema_model=False)
                    if self._use_ema_train:
                        eval_ema_epoch_loss, eval_ema_epoch_c_f1 = self.evaluate(
                        dataset_type='dev', use_ema_model=True)
                    if self.accelerator.is_local_main_process:
                        log_text = '[Evaluating] Epoch {}/{}, Step {}/{}, ' \
                            'epoch_loss:{}, epoch_c_f1:{}, '
                        logger.info(
                            log_text.format(epoch, self.epochs, step, steps, eval_epoch_loss, eval_epoch_c_f1
                                            ))
                        
                        log_text = '[Evaluating] Epoch {}/{}, Step {}/{}, ' \
                            'ema_epoch_loss:{}, ema_epoch_c_f1:{}, '
                        logger.info(
                            log_text.format(epoch, self.epochs, step, steps, eval_ema_epoch_loss, eval_ema_epoch_c_f1
                                            ))
                    if self.test_dataloader is not None:

                        test_epoch_loss, test_epoch_c_f1 = self.evaluate(
                            dataset_type='test', use_ema_model=False)
                        if self._use_ema_train:
                            test_ema_epoch_loss, test_ema_epoch_c_f1 = self.evaluate(
                            dataset_type='test', use_ema_model=True)
                        if self.accelerator.is_local_main_process:
                            log_text = '[Testing] Epoch {}/{}, Step {}/{}, ' \
                                'epoch_loss:{}, epoch_c_f1:{},'
                            logger.info(
                                log_text.format(epoch, self.epochs, step, steps,
                                                test_epoch_loss, test_epoch_c_f1))

                            log_text = '[Testing] Epoch {}/{}, Step {}/{}, ' \
                                'ema_epoch_loss:{}, ema_epoch_c_f1:{},'
                            logger.info(
                                log_text.format(epoch, self.epochs, step, steps,
                                                test_ema_epoch_loss, test_ema_epoch_c_f1))
                            
                    if self.accelerator.is_main_process and eval_epoch_c_f1 >= 0:
                        # save model
                        if eval_epoch_c_f1 >= 0:
                            max_eval_c_f1 = eval_epoch_c_f1
                            # 重置early stop次数
                            ith_early_stop_time = 0
                            final_eval_scores_for_early_stop = []
                        else:
                            # 验证集指标在下降，记录次数，为提前结束做准备。
                            ith_early_stop_time += 1
                            final_eval_scores_for_early_stop.append(
                                eval_epoch_c_f1)
                            if ith_early_stop_time >= self.early_stop_times:
                                logger.info(
                                    '[Early Stop], final eval_loss:{}'.format(
                                        eval_epoch_c_f1))
                                return
                        if self.test_dataloader is not None:
                            test_epoch_c_f1_str = str(
                                round(test_epoch_c_f1, 4))
                            
                            test_ema_epoch_c_f1_str = str(
                                round(test_ema_epoch_c_f1, 4))
                        else:
                            test_epoch_c_f1_str = 'None'
                            test_ema_epoch_c_f1_str = 'None'
                            
                        eval_epoch_c_f1_str = str(round(eval_epoch_c_f1, 4))
                        eval_ema_epoch_c_f1_str = str(round(eval_ema_epoch_c_f1, 4))
                        metric_str = 'epoch{},step{},testepochf1_{},devepochf1_{}'.format(epoch, step,
                                                                                          test_epoch_c_f1_str, eval_epoch_c_f1_str)
                        metric_mea_str = 'epoch{},step{},testepochf1_{},devepochf1_{}'.format(epoch, step,
                                                                                          test_ema_epoch_c_f1_str, eval_ema_epoch_c_f1_str)
                        saved_dir = os.path.join(
                            self.normal_model_path, metric_str)
                        self.save_model(saved_dir)
                        if self._use_ema_train:
                            saved_dir = os.path.join(self.ema_model_path, metric_mea_str)
                            self.ema_model.apply_shadow()
                            self.save_model(saved_dir)
                            self.ema_model.restore()
                        if eval_epoch_c_f1 >= 1:
                            # 验证集指标达到100%
                            logger.info(
                                'Devset loss has reached to 0, check testset f1')
                            if self.test_dataloader is not None and test_epoch_loss < 0:
                                logger.info(
                                    'Testset loss has reached to 1.0, stop training')
                                return

            if (1 / self.check_val_every_n_epoch % 1) != 0:
                epoch_end_flag = True

        return 1

    def equip_accelerator(self):

        self.model, self.optimizer, self.train_dataloader, self.dev_dataloader, self.test_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.dev_dataloader, self.test_dataloader, self.lr_scheduler)

    def train_step(self, batch_ds, ith_step):

        self.model.train()
        for k, v in batch_ds.items():
            batch_ds[k] = v.to(self.accelerator.device)

        self.optimizer.zero_grad()

        if self._use_fgm_train:
            self.fgm.attack()

        batch_c_logits, batch_d_logits, batch_loss = self.model(
            **batch_ds
        )
        batch_loss = batch_loss / self.gradient_accumulation_steps
        self.accelerator.backward(batch_loss)
        if self._use_fgm_train:
            self.fgm.restore()
        if ith_step % self.gradient_accumulation_steps == 0:
            if self._max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self._max_grad_norm)

            self.optimizer.step()
            if self._use_ema_train:
                self.ema_model.update()
            if self._warmup_steps != -1:
                self.lr_scheduler.step()

        return batch_loss.item()

    @torch.no_grad()
    def evaluate(self, dataset_type='dev', use_ema_model=False):
        # 分布式训练时, 外层调用前会确认节点为-1,0时, 才会做验证
        self.model.eval()
        if use_ema_model:
            self.ema_model.apply_shadow()
        epoch_loss, c_f1 = 0, 0
        dataloader = self.test_dataloader if dataset_type == 'test' else self.dev_dataloader
        epoch_src, epoch_gold_labels, epoch_preds = [], [], []
        for batch_ds in dataloader:
            for k, v in batch_ds.items():
                batch_ds[k] = v.to(self.accelerator.device)
            batch_c_logits, batch_d_logits, batch_loss = self.model(
                **batch_ds
            )
            
            # batch_c_logits, batch_d_logits, batch_loss, batch_src, batch_gold = self.accelerator.gather(
            #     (batch_c_logits, batch_d_logits, batch_loss, batch_ds['input_ids'], batch_ds['c_labels']))
            
            "单卡batch验证"
            batch_c_logits, batch_d_logits, batch_loss, batch_src, batch_gold = batch_c_logits, batch_d_logits, batch_loss, batch_ds['input_ids'], batch_ds['c_labels']
            try:
                epoch_loss += sum(batch_loss).item()
            except Exception as e:
                epoch_loss += batch_loss.item()
            # correct
            batch_gold = batch_gold.view(-1).cpu().numpy()
            batch_pred = torch.argmax(batch_c_logits,
                                      dim=-1).view(-1).cpu().numpy()
            batch_src = batch_src.view(-1).cpu().numpy()

            seq_true_idx = np.argwhere(
                batch_gold != self._loss_ignore_id)  # 获取非pad部分的标签

            batch_gold = batch_gold[seq_true_idx].squeeze()
            batch_pred = batch_pred[seq_true_idx].squeeze()
            batch_src = batch_src[seq_true_idx].squeeze()

            epoch_src.extend(list(batch_src))
            epoch_gold_labels.extend(list(batch_gold))
            epoch_preds.extend(list(batch_pred))

        # sentence level f1

        epoch_src = [self._keep_c_tag_id]*len(epoch_src)
        (sent_precision, sent_recall, c_f1) = f1_csc(
            src_texts=epoch_src, trg_texts=epoch_gold_labels, pred_texts=epoch_preds)
        if use_ema_model:
            self.ema_model.restore()
        return epoch_loss, c_f1
