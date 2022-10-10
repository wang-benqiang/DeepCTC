#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.trainer.trainer_gpt2 import TrainerGpt2

# 运行目录：ctc-torch
# csc 用了前20个文件
train_args = {
    'in_model_dir': 'model/gpt2_pretrain_lmdb_0207_1_2022Y07M26D17H/epoch1,step137822,testepochloss_115.86,devepochloss_16.09',  # 模型保存目录,会自动创建目录,目录名会自动加时间
    'out_model_dir': 'model/gpt2_lmdb_0228__1_pos',
    'dev_data_ratio': 0.00002,  # 如果没有dev数据，就采用比例的形式对训练集进行切分生成dev数据
    'learning_rate': 5e-5,  # 初始学习率
    'epochs': 5,
    'random_seed_num': 42,
    # 'batch_size': 78,
    # 'batch_size': 148,  #148 mixed
    'batch_size': 108,  # 148 mixed
    'max_seq_len': 128,
    'max_dataset_len': -1,  # 测试时候可以将数据集大小改为1000条进行测试, -1
    'check_val_every_n_epoch': 0.1,  # 多少个epoch进行验证
    'early_stop_times': 4,  # 如果F1连续多少个验证后不增长就停止训练
    'freeze_embedding': False,  # 是否冻结embedding层                        
    'warmup_steps': -1,
    'gradient_accumulation_steps': 1,
    'mixed_precision': 'fp16',
    'train_db_dir': '/home/ligen/api-yjy-gen-corrector/data/train_yaoge_lmdb_0228_13e/train_csc_lmdb_0228_1',
    'train_db_gb_size': 400,
    'dev_db_dir': None,
    'dev_db_gb_size': 2,
    'test_db_dir': 'data/lmdb/realise_test_ft',
    'test_db_gb_size': 22,

}


if __name__ == '__main__':

    trainer = TrainerGpt2(**train_args)
    trainer.train()
