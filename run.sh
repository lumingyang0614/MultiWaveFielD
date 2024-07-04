

python3 -u run.py --pred_len 96 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.3 --batch_size 16 --n_gnn_layer 2 --n_point 321 --data_path electricity.csv --model_id ECL_96_96 --root_path ./dataset/electricity/ --hiddenDCI 8
python3 -u run.py --pred_len 192 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.3  --batch_size 16 --n_gnn_layer 2 --n_point 321 --data_path electricity.csv --model_id ECL_96_192 --root_path ./dataset/electricity/ --hiddenDCI 8
python3 -u run.py --pred_len 336 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.1 --batch_size 16 --n_gnn_layer 2 --n_point 321 --data_path electricity.csv --model_id ECL_96_336 --root_path ./dataset/electricity/ --hiddenDCI 8
python3 -u run.py --pred_len 720 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.3 --batch_size 16 --n_gnn_layer 2 --n_point 321 --data_path electricity.csv --model_id ECL_96_720 --root_path ./dataset/electricity/ --hiddenDCI 8




python3 -u run.py --pred_len 96 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.1 --batch_size 16 --n_gnn_layer 2 --n_point 7 --data_path ETTh2.csv --model_id ETTh2_96_96 --root_path ./dataset/ETT-small/ --hiddenDCI 128
python3 -u run.py --pred_len 192 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.1  --batch_size 16 --n_gnn_layer 2 --n_point 7 --data_path ETTh2.csv --model_id ETTh2_96_192 --root_path ./dataset/ETT-small/ --hiddenDCI 128
python3 -u run.py --pred_len 336 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.1 --batch_size 16 --n_gnn_layer 2 --n_point 7 --data_path ETTh2.csv --model_id ETTh2_96_336 --root_path ./dataset/ETT-small/ --hiddenDCI 128
python3 -u run.py --pred_len 720 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.1 --batch_size 16 --n_gnn_layer 2 --n_point 7 --data_path ETTh2.csv --model_id ETTh2_96_720 --root_path ./dataset/ETT-small/ --hiddenDCI 128



python3 -u run.py --pred_len 720 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.1 --batch_size 16 --n_gnn_layer 2 --n_point 21 --data_path weather.csv --model_id weather_96_96 --root_path ./dataset/weather/ --hiddenDCI 128
python3 -u run.py --pred_len 336 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.1 --batch_size 16 --n_gnn_layer 2 --n_point 21 --data_path weather.csv --model_id weather_96_96 --root_path ./dataset/weather/ --hiddenDCI 128
python3 -u run.py --pred_len 192 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.1  --batch_size 16 --n_gnn_layer 2 --n_point 21 --data_path weather.csv --model_id weather_96_96 --root_path ./dataset/weather/ --hiddenDCI 128
python3 -u run.py --pred_len 96 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.1 --batch_size 16 --n_gnn_layer 2 --n_point 21 --data_path weather.csv --model_id weather_96_96 --root_path ./dataset/weather/ --hiddenDCI 128 


python3 -u run.py --pred_len 96 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.0 --batch_size 2 --n_gnn_layer 2 --n_point 862 --data_path traffic.csv --model_id Traffic_96_96 --root_path ./dataset/traffic/ --hiddenDCI 2
python3 -u run.py --pred_len 192 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.0  --batch_size 2 --n_gnn_layer 2 --n_point 862 --data_path traffic.csv --model_id Traffic_96_192 --root_path ./dataset/traffic/ --hiddenDCI 2
python3 -u run.py --pred_len 336 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.0 --batch_size 2 --n_gnn_layer 2 --n_point 862 --data_path traffic.csv --model_id Traffic_96_336 --root_path ./dataset/traffic/ --hiddenDCI 2
python3 -u run.py --pred_len 720 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.0 --batch_size 2 --n_gnn_layer 2 --n_point 862 --data_path traffic.csv --model_id Traffic_96_720 --root_path ./dataset/traffic/ --hiddenDCI 2









python3 -u run.py --pred_len 12 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.1  --batch_size 16 --n_gnn_layer 2 --seq_len 144 --n_point 137 --data_path solar_AL.csv --model_id solar_144_24 --root_path ./dataset/solar-energy/ --hiddenDCI 16
python3 -u run.py --pred_len 12 --learning_rate 1e-3 --wavelet_j 3 --seed 4321 --drop 0.1  --batch_size 16 --n_gnn_layer 2 --seq_len 144 --n_point 137 --data_path solar_AL.csv  --model_id solar_144_12 --root_path ./dataset/solar-energy/ --hiddenDCI 16
python3 -u run.py --pred_len 6 --learning_rate 1e-3 --wavelet_j 2 --seed 4321 --drop 0.1  --batch_size 16 --n_gnn_layer 2 --seq_len 144 --n_point 137 --data_path solar_AL.csv --model_id solar_144_6 --root_path ./dataset/solar-energy/ --hiddenDCI 16

