cd ..
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch -m runner.train_ctc_t5