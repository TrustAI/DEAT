import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=200, type=int)
parser.add_argument('--model-dir', default='mdeat_out', type=str)
parser.add_argument('--log-name', default='aa_score', type=str)
args = parser.parse_args()
log_path = os.path.join(args.model_dir,args.log_name+'.log')
nb_batch = 0
nb_robust_examples = 0
nb_correct_examples = 0
with open(log_path) as log_reader:
    l = log_reader.readline()
    while l:
        if "robust accuracy:" in l:
            nb_robust_examples += int(float(l.split(':')[1].strip()[:-1])*2)
            nb_batch += 1
        elif "initial accuracy:" in l:
            nb_correct_examples += int(float(l.split(':')[1].strip()[:-1])*2)

        l = log_reader.readline()
print( f"{args.model_dir+args.log_name}: AA score is {nb_robust_examples/(nb_batch * args.batch_size) * 100:.2f}!")
print( f"{args.model_dir+args.log_name}: Clean acc is {nb_correct_examples/(nb_batch * args.batch_size) * 100:.2f}!")
