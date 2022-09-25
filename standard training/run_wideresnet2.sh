cuda='3'
model='wide'
model_name='model_'$model
# methods_list1='
# mdeat
# mpgd
# '

# for method in $methods_list1; do
#     train_method=${method}"_cifar10.py"
#     out_dir=${method}"_out"

#     CUDA_VISIBLE_DEVICES=$cuda python $train_method --alpha 10 --lr-max 0.1 --model $model --out-dir $out_dir --fname 'output_'$model --epoch 100 --save-model
#     CUDA_VISIBLE_DEVICES=$cuda python autoattack_cifar10.py --model $model --model-dir $out_dir --model-name $model_name --log-name 'aa_score_'$model
# done

# methods_list_0='
# amata
# pgd
# '

# for method in $methods_list_0; do
#     train_method=${method}"_cifar10.py"
#     out_dir=${method}"_out"

#     CUDA_VISIBLE_DEVICES=$cuda python $train_method --lr-max 0.1 --model $model --alpha 2 --out-dir $out_dir --fname 'output_'$model --epoch 100 --save-model
#     CUDA_VISIBLE_DEVICES=$cuda python autoattack_cifar10.py --model $model --model-dir $out_dir --model-name $model_name --log-name 'aa_score_'$model
# done

# mtrades
methods_list2='
mmart
'
# trades
# mart
# fat_mart
# fat
# fat_trade

for method in $methods_list2; do
    train_method=${method}"_cifar10.py"
    out_dir=${method}"_out"

    CUDA_VISIBLE_DEVICES=$cuda python $train_method --lr 0.1 --model $model --step-size 2 --out-dir $out_dir --fname 'output_'$model  --epoch 100 --save-model
    CUDA_VISIBLE_DEVICES=$cuda python autoattack_cifar10.py --model $model --model-dir $out_dir --model-name $model_name --log-name 'aa_score_'$model
done
