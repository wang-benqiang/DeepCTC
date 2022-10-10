#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.trainer.trainer_csc_seq2edit import TrainerCscSeq2Edit

# 运行目录：ctc-torch
# csc 用了前20个文件
train_args = {
    'in_model_dir': 'model/csc_s2e_finetune3_2022Y07M28D10H/epoch4,step88,testepochf1_0.9868,devepochf1_0.9931',  # 模型保存目录,会自动创建目录,目录名会自动加时间
    'out_model_dir': 'model/csc_s2e_ccl_finetune_final2',
    'dev_data_ratio': 0.05,  # 如果没有dev数据，就采用比例的形式对训练集进行切分生成dev数据
    'learning_rate': 5e-5,  # 初始学习率
    'epochs': 7,
    'random_seed_num': 897,
    # 'batch_size': 78,
    'batch_size': 152,  #148 mixed
    'max_seq_len': 128,
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

    trainer = TrainerCscSeq2Edit(**train_args)
    trainer.train()
