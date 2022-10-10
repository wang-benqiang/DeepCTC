cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m runner.train_csc_seq2edit