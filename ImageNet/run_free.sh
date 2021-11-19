CONFIG=configs/configs_free.yml
DATA=/datasets/ImageNet2012

CUDA_VISIBLE_DEVICES=1,2,3 python free_imagenet.py $DATA -c $CONFIG