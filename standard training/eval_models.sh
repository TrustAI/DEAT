model='dense'
model_name='model_dense'
cuda='0'

# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir mdeat_out   --model $model --model-name $model_name >> ./robust_test.txt
# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir mtrades_out --model $model --model-name $model_name >> ./robust_test.txt
# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir mmart_out   --model $model --model-name $model_name >> ./robust_test.txt
# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir mpgd_out    --model $model --model-name $model_name >> ./robust_test.txt
# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir free_out    --model $model --model-name $model_name'4' >> ./robust_test.txt
# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir free_out    --model $model --model-name $model_name'6' >> ./robust_test.txt
# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir free_out    --model $model --model-name $model_name'8' >> ./robust_test.txt
# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir pgd_out     --model $model --model-name $model_name >> ./robust_test.txt
# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir mart_out    --model $model --model-name $model_name >> ./robust_test.txt
CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir trades_out  --model $model --model-name $model_name >> ./robust_test.txt
# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir fat_out     --model $model --model-name $model_name >> ./robust_test.txt
# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir fat_mart_out  --model $model --model-name $model_name >> ./robust_test.txt
# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir fat_trade_out --model $model --model-name $model_name >> ./robust_test.txt
# CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model-dir amata_out     --model $model --model-name $model_name >> ./robust_test.txt