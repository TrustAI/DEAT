cuda='1'
model='vgg16'
model_name='model_'$model
# free
methods_list1='
mdeat
mpgd
amata
pgd
'
for method in $methods_list1; do
    train_method=${method}"_cifar10.py"
    out_dir=${method}"_out"

    CUDA_VISIBLE_DEVICES=$cuda python $train_method --lr-max 0.01 --model $model --out-dir $out_dir --fname 'output_'$model --save-model
    CUDA_VISIBLE_DEVICES=$cuda python autoattack_cifar10.py --model $model --model-dir $out_dir --model-name $model_name --log-name 'aa_score_'$model
done

methods_list2='
fat_mart
fat
fat_trade
mart
mmart
mtrades
trades
'

for method in $methods_list2; do
    train_method=${method}"_cifar10.py"
    out_dir=${method}"_out"

    CUDA_VISIBLE_DEVICES=$cuda python $train_method --lr 0.01 --model $model --out-dir $out_dir --fname 'output_'$model --save-model
    CUDA_VISIBLE_DEVICES=$cuda python autoattack_cifar10.py --model $model --model-dir $out_dir --model-name $model_name --log-name 'aa_score_'$model
done