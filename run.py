import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import torch.utils.checkpoint as checkpoint
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True

torch.backends.cudnn.enabled = False
parser = argparse.ArgumentParser()
fix_seed = 2021

random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
import wandb
wandb.login()
parser.add_argument("--model_id", type=str, default="solar_96_96", help="model id")
parser.add_argument(
    "--model",
    type=str,
    default="WaveForM",
    help="model name, options: [WaveForM]",
)

parser.add_argument("--data", type=str, default="custom", help="dataset type")
parser.add_argument(
    "--root_path",
    type=str,
    # default="./dataset/electricity/",
    # default="./dataset/weather/",
    # default="./dataset/ETT-small/",
    # default="./dataset/traffic/",
    default="./dataset/solar-energy/",
    help="root path of the data file",
)
parser.add_argument(
    # "--data_path", type=str, default="electricity.csv", help="data file"
    # "--data_path", type=str, default="weather.csv", help="data file"
    # "--data_path", type=str, default="ETTh2.csv", help="data file"
    # "--data_path", type=str, default="traffic.csv", help="data file"
    "--data_path", type=str, default="solar_AL.csv", help="data file"

)
parser.add_argument(
    "--checkpoints",
    type=str,
    default="./checkpoints/",
    help="location of model checkpoints",
)

parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
parser.add_argument(
    "--pred_len", type=int, default=336, help="prediction sequence length"
)

# parser.add_argument("--n_points", type=int, default=21, help="the number of variables")
# parser.add_argument("--n_points", type=int, default=7, help="the number of variables")
# parser.add_argument("--n_points", type=int, default=321, help="the number of variables")
# parser.add_argument("--n_points", type=int, default=862, help="the number of variables")
parser.add_argument("--n_points", type=int, default=137, help="the number of variables")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout")

parser.add_argument("--itr", type=int, default=1, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size of train input data"
)
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument(
    "--learning_rate", type=float, default=0.0003, help="optimizer learning rate"
)
parser.add_argument("--des", type=str, default="Exp", help="exp description")
parser.add_argument("--loss", type=str, default="mse", help="loss function")
parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")

parser.add_argument("--node_dim", type=int, default=2, help="node_dim in graph")
parser.add_argument("--subgraph_size", type=int, default=1, help="the subgraph size, i.e. topk")
parser.add_argument("--n_gnn_layer", type=int, default=3, help="number of layers in GNN.")
parser.add_argument("--wavelet_j", type=int, default=3, help="the number of wavelet layer")
parser.add_argument("--wavelet", type=str, default='haar', help='the wavelet function')
parser.add_argument('--dish_init', type=str, default='uniform') # standard, 'avg' or 'uniform'
parser.add_argument('--norm', type=str, default='none') # none, revin, dishts
parser.add_argument('--seed', type=str, default='4321') # none, revin, dishts
parser.add_argument('--hiddenDCI', type=int, default='1') # none, revin, dishts
args = parser.parse_args()

torch.manual_seed(args.seed)  # reproducible
torch.cuda.manual_seed_all(args.seed)
wandb.init(project="WaveForm_Traffic"+str(args.pred_len), name="All"+args.seed +'_drop_'+str(args.dropout)+'lr'+str(args.learning_rate))


print("Args in experiment:")
import json

print(json.dumps(vars(args), indent=4, ensure_ascii=False))

Exp = Exp_Main

from utils import color
for ii in range(args.itr):
    setting = f"{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_{args.des}_{ii}"
    
    exp = Exp(args)
    color.cprint(f'start training:\n{setting}', color.OKGREEN, end='\n')
    
    exp.train(setting)
    
    color.cprint(f'end of training. begin testing', color.OKGREEN, '\n')
    exp.test(setting)
