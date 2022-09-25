cuda='3'
model='pre'
root='./new_results/'
model_name='model_'$model
methods_list1='
mpgd
mtrades
mmart
mdeat
'

for method in $methods_list1; do
    train_method=${method}"_cifar10.py"
    out_dir=${root}${method}"_out"

    CUDA_VISIBLE_DEVICES=$cuda python $train_method --alpha 2 --lr-max 0.05 --epochs 5 --model $model --out-dir $out_dir --fname 'output_'$model #--save-model
#     CUDA_VISIBLE_DEVICES=$cuda python autoattack_cifar10.py --model $model --model-dir $out_dir --model-name $model_name --log-name 'aa_score_'$model
done

# methods_list_0='
# amata
# pgd
# '

# for method in $methods_list_0; do
#     train_method=${method}"_cifar10.py"
#     out_dir=${method}"_out"

#     CUDA_VISIBLE_DEVICES=$cuda python $train_method --lr-max 0.01 --model $model --alpha 2 --out-dir $out_dir --fname 'output_'$model --save-model
#     CUDA_VISIBLE_DEVICES=$cuda python autoattack_cifar10.py --model $model --model-dir $out_dir --model-name $model_name --log-name 'aa_score_'$model
# done

# mtrades
# mmart
# fat_mart
# fat
# fat_trade
# mart
# methods_list2='
# trades
# '

# for method in $methods_list2; do
#     train_method=${method}"_cifar10.py"
#     out_dir=${method}"_out"

#     CUDA_VISIBLE_DEVICES=$cuda python $train_method --lr 0.01 --model $model --step_size 2 --out-dir $out_dir --fname 'output_'$model --save-model
#     CUDA_VISIBLE_DEVICES=$cuda python autoattack_cifar10.py --model $model --model-dir $out_dir --model-name $model_name --log-name 'aa_score_'$model
# done
