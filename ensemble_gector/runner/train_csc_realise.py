#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.trainer.trainer_csc_realise import TrainerCscRealise

# 运行目录：ctc-torch
# csc 用了前20个文件
train_args = {
    'in_model_dir': 'model/realise_ccl2022_2022Y07M24D23H/epoch4,ith_db:0,step75,testf1_99_57%,devf1_99_62%',  # 模型保存目录,会自动创建目录,目录名会自动加时间
    'out_model_dir': 'model/csc_realise_ccl_test_ft2',
    'dev_data_ratio': 0.05,  # 如果没有dev数据，就采用比例的形式对训练集进行切分生成dev数据
    'learning_rate': 5e-5,  # 初始学习率
    'epochs': 5,
    'random_seed_num': 42,
    # 'batch_size': 78,
    'batch_size': 128,  #148 mixed
    'max_seq_len': 78,
    'max_dataset_len': -1,  # 测试时候可以将数据集大小改为1000条进行测试
    'check_val_every_n_epoch': 1,  # 多少个epoch进行验证
    'early_stop_times': 4,  # 如果F1连续多少个验证后不增长就停止训练
    'freeze_embedding': False,  # 是否冻结embedding层                        
    'warmup_steps': 10,
    'gradient_accumulation_steps': 1,
    'mixed_precision': 'fp16',
    'train_db_dir': 'data/lmdb/track1test_artificial_new',
    'train_db_gb_size': 2,
    'dev_db_dir': None,
    'dev_db_gb_size': 2,
    'test_db_dir': 'data/lmdb/realise_test_ft',
    'test_db_gb_size': 2,
    'use_ema_train': False

}


if __name__ == '__main__':

    trainer = TrainerCscRealise(**train_args)
    trainer.train()
