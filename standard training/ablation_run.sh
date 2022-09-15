# CUDA_VISIBLE_DEVICES=2 python mdeat_cifar10.py --gamma 1.01 --epochs 50 --model pre --save-model --lr-schedule cyclic --out-dir ablation_cyclic_out 
# CUDA_VISIBLE_DEVICES=2 python mdeat_cifar10.py --gamma 1.01 --epochs 50 --model pre --save-model --lr-schedule flat --out-dir ablation_flat_out 
# CUDA_VISIBLE_DEVICES=2 python mdeat_cifar10.py --gamma 1.01 --epochs 50 --model pre --save-model --delta-init zero --out-dir ablation_zero_out 
# CUDA_VISIBLE_DEVICES=2 python mdeat_cifar10.py --gamma 1.01 --epochs 50 --model pre --save-model --delta-init normal --out-dir ablation_normal_out 
# CUDA_VISIBLE_DEVICES=2 python mdeat_cifar10.py --gamma 1.01 --alpha 8 --epochs 50 --model pre --save-model --out-dir ablation_8_out 
# CUDA_VISIBLE_DEVICES=2 python mdeat_cifar10.py --gamma 1.01 --alpha 12 --epochs 50 --model pre --save-model --out-dir ablation_12_out 


CUDA_VISIBLE_DEVICES=0 python evaluation_cifar10.py --model-dir ablation_12_out   --model pre --model-name model_pre >> ./ablation_result.txt
CUDA_VISIBLE_DEVICES=0 python evaluation_cifar10.py --model-dir ablation_8_out   --model pre --model-name model_pre >> ./ablation_result.txt
CUDA_VISIBLE_DEVICES=0 python evaluation_cifar10.py --model-dir ablation_normal_out   --model pre --model-name model_pre >> ./ablation_result.txt
CUDA_VISIBLE_DEVICES=0 python evaluation_cifar10.py --model-dir ablation_zero_out   --model pre --model-name model_pre >> ./ablation_result.txt
CUDA_VISIBLE_DEVICES=0 python evaluation_cifar10.py --model-dir ablation_flat_out   --model pre --model-name model_pre >> ./ablation_result.txt
CUDA_VISIBLE_DEVICES=0 python evaluation_cifar10.py --model-dir ablation_cyclic_out   --model pre --model-name model_pre >> ./ablation_result.txt