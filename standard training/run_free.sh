cuda='3'
model='vgg'
model_name='model_'$model

batch_replay='
4
6
8
'
for br in $batch_replay; do
    CUDA_VISIBLE_DEVICES=$cuda python free_cifar10.py --lr-max 0.01 --max-iteration $br --model $model --out-dir free_out --fname 'output_'$model$br --save-model
    CUDA_VISIBLE_DEVICES=$cuda python autoattack_cifar10.py --model $model --model-dir free_out --model-name $model_name$br --log-name 'aa_score_'$model$br
done