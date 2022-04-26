cuda='1'
model='vgg'
model_name='model_'$model
# free
methods_list='
amata
pgd
fat
fat_trade
trades
mdeat
mart
mmart
mpgd
mtrades
'
# fat_mart


for method in $methods_list; do
    # train_method=${method}"_cifar10.py"
    out_dir=${method}"_out"

    # CUDA_VISIBLE_DEVICES=$cuda python $train_method --model $model --out-dir $out_dir --fname 'output_'$model --save-model
    CUDA_VISIBLE_DEVICES=$cuda python evaluation_cifar10.py --model $model --model-dir $out_dir --model-name $model_name >> robust_test.txt
done