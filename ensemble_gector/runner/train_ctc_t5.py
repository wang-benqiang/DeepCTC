#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.trainer.trainer_ctc_t5 import TrainerCtcT5

# 运行目录：ctc-torch
# csc 用了前20个文件
train_args = {
    'in_model_dir': 'model/gec_t5_track2_pretrained_new_2022Y08M21D15H/epoch8,step3385,testepochf1_0.1457,devepochf1_0.1349',  # 模型保存目录,会自动创建目录,目录名会自动加时间
    # 'out_model_dir': 'model/gec_t5_train_lang8',
    'out_model_dir': 'model/gec_t5_track2_finetune_cged',
    'dev_data_ratio': 0.001,  # 如果没有dev数据，就采用比例的形式对训练集进行切分生成dev数据
    'learning_rate': 2e-4,  # 初始学习率
    'epochs': 10,
    'random_seed_num': 42,
    'batch_size': 32,  # 48, 256
    # 'batch_size': 4,  #148 mixed
    'max_seq_len': 128,
    'max_dataset_len': -1,  # 测试时候可以将数据集大小改为1000条进行测试
    'check_val_every_n_epoch': 1,  # 多少个epoch进行验证
    'early_stop_times': 10,  # 如果F1连续多少个验证后不增长就停止训练
    'freeze_embedding': False,  # 是否冻结embedding层       
    'freeze_encoder': False,                 
    'freeze_backbone': False,                 
    'warmup_steps': 200,
    'gradient_accumulation_steps': 4,
    'mixed_precision': 'fp16',
    'train_db_dir': 'data/lmdb/track2_cged_train',
    # 'train_db_dir': 'data/lmdb/artificial_data_new_ccl2022',
    'train_db_gb_size': 2,
    'dev_db_dir': None,
    'dev_db_gb_size': 2,
    'test_db_dir': 'data/lmdb/cged2021_test',
    'test_db_gb_size': 2,
    'use_ema_train': False,

}


if __name__ == '__main__':

    trainer = TrainerCtcT5(**train_args)
    trainer.train()
