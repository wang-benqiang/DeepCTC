#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.trainer.trainer_mlm_bert import TrainerMlmBert

# 运行目录：ctc-torch
# csc 用了前20个文件
train_args = {
    # 模型保存目录,会自动创建目录,目录名会自动加时间
    'in_model_dir': 'pretrained_model/roberta_10e_mlm/pretain_new_2022Y09M19D11H/epoch1,step154070,testepochf1_0.6869,devepochf1_0.7806',
    'out_model_dir': 'pretrained_model/roberta_10e_mlm/pretain_new',
    'dev_data_ratio': 0.0001,  # 如果没有dev数据，就采用比例的形式对训练集进行切分生成dev数据
    'learning_rate': 1e-4,  # 初始学习率
    'epochs': 1,
    'random_seed_num': 42,
    # 'batch_size': 78,
    'batch_size': 50,  # 148 mixed
    'max_seq_len': 128,
    'max_dataset_len': -1,  # 测试时候可以将数据集大小改为1000条进行测试
    'check_val_every_n_epoch': 0.1,  # 多少个epoch进行验证
    'early_stop_times': 10,  # 如果F1连续多少个验证后不增长就停止训练
    'freeze_embedding': False,  # 是否冻结embedding层
    'freeze_backbone': False,  # 是否冻结embedding层
    'warmup_steps': 2500,
    'gradient_accumulation_steps': 5,
    'mixed_precision': 'fp16',
    'train_db_dir': 'data/lmdb/train_csc_lg_0228_13e/train_csc_lmdb_2',
    'train_db_gb_size': 2,
    'dev_db_dir': None,
    'dev_db_gb_size': 2,
    'test_db_dir': 'data/lmdb/realise_test_ft',
    'test_db_gb_size': 2,
    'use_ema_train': True,
    'use_fgm_train': False,

}


if __name__ == '__main__':

    trainer = TrainerMlmBert(**train_args)
    trainer.train()
