#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.trainer.trainer_ctc_seq2edit import TrainerCtcSeq2Edit

# 运行目录：ctc-torch
# csc 用了前20个文件
train_args = {
    # 模型保存目录,会自动创建目录,目录名会自动加时间
    'in_model_dir': 'model/gector_finetune2_2022Y08M22D10H/epoch10,step148,testepochf1_0.3166,devepochf1_0.5452',
    'out_model_dir': 'model/gector/finetune_final1', # finetune_final1
    'dev_data_ratio': 0.005,  # 如果没有dev数据，就采用比例的形式对训练集进行切分生成dev数据
    'learning_rate': 5e-5,  # 初始学习率
    'epochs': 10,
    'random_seed_num': 1234,
    # 'batch_size': 78,
    'batch_size': 48,  # 148 mixed
    'max_seq_len': 128,
    'max_dataset_len': -1,  # 测试时候可以将数据集大小改为1000条进行测试
    'check_val_every_n_epoch': 1,  # 多少个epoch进行验证
    'early_stop_times': 4,  # 如果F1连续多少个验证后不增长就停止训练
    'freeze_embedding': False,  # 是否冻结embedding层
    'freeze_backbone': False,  # 是否冻结embedding层
    # 'freeze_backbone_in_first_n_epochs': 2,
    'warmup_steps': 200,
    'gradient_accumulation_steps': 2,
    'mixed_precision': 'fp16',
    'train_db_dir': 'data/lmdb/cged_sighan_10pos_train',
    'train_db_gb_size': 2,
    'dev_db_dir': None,
    'dev_db_gb_size': 2,
    'test_db_dir': 'data/lmdb/cged2021+100pos',
    'test_db_gb_size': 2,
    'use_ema_train': True,
    'use_fgm_train': False,
    

}


if __name__ == '__main__':

    trainer = TrainerCtcSeq2Edit(**train_args)
    trainer.train()
