CONFIG=configs/configs_mdeat.yml
DATA=/datasets/ImageNet2012

CUDA_VISIBLE_DEVICES=1,2,3 python mdeat_imagenet.py $DATA -c $CONFIG