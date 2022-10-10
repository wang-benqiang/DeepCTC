cd ..
CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch -m runner.train_ctc_seq2edit