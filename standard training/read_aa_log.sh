log_name='aa_score_vgg16'

python aalog_reader.py --model-dir mdeat_out --log-name $log_name
python aalog_reader.py --model-dir mtrades_out --log-name $log_name
python aalog_reader.py --model-dir mmart_out --log-name $log_name
python aalog_reader.py --model-dir mpgd_out --log-name $log_name
python aalog_reader.py --model-dir free_out --log-name $log_name'4'
python aalog_reader.py --model-dir free_out --log-name $log_name'6'
python aalog_reader.py --model-dir free_out --log-name $log_name'8'
python aalog_reader.py --model-dir pgd_out --log-name $log_name
python aalog_reader.py --model-dir mart_out --log-name $log_name
python aalog_reader.py --model-dir trades_out --log-name $log_name
python aalog_reader.py --model-dir fat_out --log-name $log_name
python aalog_reader.py --model-dir fat_mart_out --log-name $log_name
python aalog_reader.py --model-dir fat_trade_out --log-name $log_name
python aalog_reader.py --model-dir amata_out --log-name $log_name