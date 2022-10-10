#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import json
import os
import random
import time
from curses.ascii import EM
from math import ceil
from typing import List, Optional

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from logs import logger
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn, track)
from src.dataset.dataset_gec_t5 import DatasetGecT5
from src.metrics.metric_csc import f1_csc
from src.modeling.modeling_ctc_t5 import ModelingCtcT5, T5TokenizerFast
from src.tricks.ema_torch import EMA
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


class TrainerCtcT5:
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
                 freeze_encoder: bool = False,
                 freeze_backbone: bool = False,
                 warmup_steps: int = -1,
                 max_grad_norm: Optional[float] = None,
                 dev_data_ratio: Optional[float] = 0.2,
                 loss_ignore_id=-100,
                 mixed_precision='fp16',
                 ctc_label_vocab_dir: str = 'src/vocab',
                 gradient_accumulation_steps: Optional[int] = 1,
                 use_ema_train=False,
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
        import inspect
        self.in_model_dir = in_model_dir
        self.out_model_dir = os.path.join(out_model_dir, '')[
            :-1] + current_time

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_len = max_seq_len
        self.random_seed_num = random_seed_num
        self.freeze_embedding = freeze_embedding
        self.freeze_encoder = freeze_encoder
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if not os.path.exists(self.out_model_dir) and self.accelerator.is_main_process:
            os.mkdir(self.out_model_dir)

        # load data, model, scheduler
        self.fit_seed(self.random_seed_num)
        self.tokenizer = T5TokenizerFast.from_pretrained(self.in_model_dir)
        self.train_dataloader, self.dev_dataloader, self.test_dataloader = self.load_data()
        self.model, self.optimizer, self.lr_scheduler = self.load_suite()
        
        self.equip_accelerator()
        
        if self._use_ema_train:
            self.ema_model = EMA(self.model, 0.9999)
            self.ema_model.register()

        # save train params
        if self.accelerator.is_main_process:
            signature = inspect.signature(TrainerCtcT5)
            train_kwargs = {}
            for param in signature.parameters.values():
                train_kwargs[param.name] = eval(param.name)
            train_kwargs = {k: v for k, v in train_kwargs.items(
            ) if isinstance(v, (str, bool, float, int))}

            out_fp = open(os.path.join(self.out_model_dir, 'train_config.json'),
                        'w', encoding='utf8')
            json.dump(train_kwargs, out_fp, ensure_ascii=False, indent=4)  
            out_fp.close()
    
    def load_data(self) -> List[DataLoader]:

        # 加载train-dataset

        train_ds = DatasetGecT5(
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            db_dir=self.train_db_dir,
            max_dataset_len=self.max_dataset_len
        )

        if self.dev_db_dir is not None and os.path.exists(self.dev_db_dir):
            dev_ds = DatasetGecT5(
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
            test_ds = DatasetGecT5(
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                db_dir=self.test_db_dir,
                max_dataset_len=self.max_dataset_len
            )

        else:
            test_ds = None

        train_dataloader = torch.utils.data.dataloader.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8, collate_fn=test_ds.collate_fn)
        dev_dataloader = torch.utils.data.dataloader.DataLoader(
            dev_ds, batch_size=self.batch_size, shuffle=False, num_workers=8, collate_fn=test_ds.collate_fn)
        if test_ds is not None:
            test_dataloader = torch.utils.data.dataloader.DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=False, num_workers=8, collate_fn=test_ds.collate_fn)

        return [train_dataloader, dev_dataloader, test_dataloader]

    def load_suite(self):
        "model"
        model = ModelingCtcT5.from_pretrained(
            self.in_model_dir)

        model._init_criterion()
        if self.freeze_embedding:
            embedding_name_list = ('embeddings.word_embeddings.weight',
                                   'embeddings.position_embeddings.weight',
                                   'embeddings.token_type_embeddings.weight')
            for named_para in model.named_parameters():
                named_para[1].requires_grad = False if named_para[
                    0] in embedding_name_list else True

        if self.freeze_encoder:
            for named_para in model.named_parameters():
                named_para[1].requires_grad = False if 'encoder' in named_para[
                    0] else True

        if self.freeze_backbone:
            for named_para in model.named_parameters():
                named_para[1].requires_grad = False if 'lm_head' not in named_para[
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

    def load_progress_bar(self):

        return Progress(TimeRemainingColumn(),
                        TimeElapsedColumn(),
                        BarColumn(),
                        TextColumn(
                            "[progress.percentage]{task.percentage:>3.0f}%"),
                        TextColumn("[progress.description]{task.description}"),
                        disable=True if not self.accelerator.is_main_process else False  # support ddp mode
                        )

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
        "load progress bar"
        self.progress_bar = self.load_progress_bar()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            with self.progress_bar:
                if self.accelerator.is_main_process:
                    steps_progress_bar = self.progress_bar.add_task(
                        description='', total=self._train_steps)
                for step, batch_ds in enumerate(self.train_dataloader):
                    step += 1
                    overall_step += 1
                    batch_loss = self.train_step(
                        batch_ds, ith_step=step)

                    epoch_loss += batch_loss
                    if self.accelerator.is_main_process and (step % ceil(self.check_val_every_n_epoch * len(self.train_dataloader)) == 0 or epoch_end_flag):
                        #  到达验证步数，则开始验证，保存模型，记录最大的dev指标
                        logger.info('[Start Evaluating]')
                        epoch_end_flag = False
                        eval_epoch_loss, eval_epoch_c_f1 = self.evaluate(
                            dataset_type='dev')

                        if self.accelerator.is_local_main_process:
                            log_text = '[Evaluating] Epoch {}/{}, Step {}/{}, ' \
                                'epoch_loss:{}, epoch_c_f1:{}, '
                            logger.info(
                                log_text.format(epoch, self.epochs, step, steps, eval_epoch_loss, eval_epoch_c_f1
                                                ))
                        if self.test_dataloader is not None:

                            test_epoch_loss, test_epoch_c_f1 = self.evaluate(
                                dataset_type='test')
                            if self.accelerator.is_local_main_process:
                                log_text = '[Testing] Epoch {}/{}, Step {}/{}, ' \
                                    'epoch_loss:{}, epoch_c_f1:{},'
                                logger.info(
                                    log_text.format(epoch, self.epochs, step, steps,
                                                    test_epoch_loss, test_epoch_c_f1))

                        if self.accelerator.is_main_process and eval_epoch_c_f1 >= 0:
                            # save model
                            if eval_epoch_c_f1 >= max_eval_c_f1:
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
                            else:
                                test_epoch_c_f1_str = 'None'
                            eval_epoch_c_f1_str = str(
                                round(eval_epoch_c_f1, 4))
                            metric_str = 'epoch{},step{},testepochf1_{},devepochf1_{}'.format(epoch, step,
                                                                                              test_epoch_c_f1_str, eval_epoch_c_f1_str)
                            saved_dir = os.path.join(
                                self.out_model_dir, metric_str)

                            if self._use_ema_train:
                                self.ema_model.apply_shadow()
                            self.save_model(saved_dir)
                            if self._use_ema_train:
                                self.ema_model.restore()
                            if eval_epoch_c_f1 >= 1:
                                # 验证集指标达到100%
                                logger.info(
                                    'Devset loss has reached to 0, check testset f1')
                                if self.test_dataloader is not None and test_epoch_loss < 0:
                                    logger.info(
                                        'Testset loss has reached to 1.0, stop training')
                                    return
                    if self.accelerator.is_main_process:
                        self.progress_bar.update(steps_progress_bar, description='[Training] [{}/{} Steps, {}/{} Epochs, Batch Loss:{}]'.format(
                            step, self._train_steps, epoch, self.epochs, round(batch_loss, 4)), advance=1)
                if (1 / self.check_val_every_n_epoch % 1) != 0:
                    epoch_end_flag = True
                    # self.progress_bar.reset(steps_progress_bar)
                    self.progress_bar.remove_task(steps_progress_bar)
        return 1

    def equip_accelerator(self):

        self.model, self.optimizer, self.train_dataloader,  self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler)

    def train_step(self, batch_ds, ith_step):

        self.model.train()
        for k, v in batch_ds.items():
            batch_ds[k] = v.to(self.accelerator.device)

        self.optimizer.zero_grad()

        # 常规模式
        batch_loss = self.model(**batch_ds).loss
        batch_loss = batch_loss / self.gradient_accumulation_steps
        self.accelerator.backward(batch_loss)
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
    def evaluate(self, dataset_type='dev'):
        # 分布式训练时, 外层调用前会确认节点为-1,0时, 才会做验证
        self.model.eval()
        if self._use_ema_train:
            self.ema_model.apply_shadow()
        epoch_loss, sent_f1 = 0, 0
        dataloader = self.test_dataloader if dataset_type == 'test' else self.dev_dataloader
        epoch_src, epoch_gold_labels, epoch_preds = [], [], []

        steps = len(dataloader)
        if self.accelerator.is_main_process:
            eval_steps_progress_bar = self.progress_bar.add_task(
                description='[Evaluation]', total=steps)
        for step, batch_ds in enumerate(dataloader):
            step += 1
            for k, v in batch_ds.items():
                batch_ds[k] = v.to(self.accelerator.device)

            try:
                # Fix bug with
                # AttributeError: 'DistributedDataParallel' object has no attribute 'generate'
                batch_c_logits = self.model.module.generate(
                    input_ids=batch_ds['input_ids'], max_length=128
                )
            except:
                batch_c_logits = self.model.generate(
                    input_ids=batch_ds['input_ids'], max_length=128
                )
            batch_src, batch_gold = batch_ds['input_ids'], batch_ds['labels']
            # batch_c_logits, batch_src, batch_gold = self.accelerator.gather(
            #     (batch_c_logits, batch_ds['input_ids'], batch_ds['labels']))
            # correct
            # if self.accelerator.is_main_process:
            epoch_src += self.tokenizer.batch_decode(
                batch_src, skip_special_tokens=True)
            epoch_gold_labels += self.tokenizer.batch_decode(torch.where(
                batch_gold == -100, 0, batch_gold), skip_special_tokens=True)
            epoch_preds += self.tokenizer.batch_decode(
                batch_c_logits, skip_special_tokens=True)
            if self.accelerator.is_main_process:
                self.progress_bar.update(
                    eval_steps_progress_bar, description='[Evaluation {}] [{}/{} Steps]'.format(dataset_type, step, steps), advance=1)

        epoch_src = [i.replace(' ', '') for i in epoch_src]
        epoch_gold_labels = [i.replace(' ', '') for i in epoch_gold_labels]
        epoch_preds = [i.replace(' ', '') for i in epoch_preds]

        (sent_precision, sent_recall, sent_f1) = f1_csc(
            src_texts=epoch_src, trg_texts=epoch_gold_labels, pred_texts=epoch_preds)

        if self._use_ema_train:
            self.ema_model.restore()
        self.progress_bar.remove_task(eval_steps_progress_bar,)
        return epoch_loss, sent_f1
