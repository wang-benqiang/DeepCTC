cd ..
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch -m runner.train_mlm_bert