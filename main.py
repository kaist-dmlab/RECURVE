from eval import *
from measures import *
from train import *
from utils import *
import argparse
import pandas as pd
import os

input_dims_dict = {"50salads":2048, "HAPT":6, "mHealth": 23, "WISDM": 3}
num_class_dict = {"50salads":19, "HAPT":6, "mHealth": 12, "WISDM": 6}
sampling_rate_dict = {"50salads":30, "HAPT":50, "mHealth": 50, "WISDM": 20} 
mean_segment_length = {"50salads":551, "HAPT":903, "mHealth": 2932, "WISDM": 697} 
data_classlabel_dict = {
    "WISDM": {0:"walk",1:"jog",2:"stair",3:"sit",4:"stand",5:"lie"},
    "HAPT": {0:"walk",1:"up",2:"down",3:"sit",4:"stand",5:"lie"},
    "50salads": {0 :"cut_tomato", 1 :"place_tomato_into_bowl", 2 :"cut_cheese", 3 :"place_cheese_into_bowl", 4 :"cut_lettuce", 5 :"place_lettuce_into_bowl", 6 :"add_salt", 7 :"add_vinegar", 8 :"add_oil", 9 :"add_pepper", 10: "mix_dressing", 11: "peel_cucumber", 12: "cut_cucumber", 13: "place_cucumber_into_bowl", 14: "add_dressing", 15: "mix_ingredients", 16: "serve_salad_onto_plate", 17: "action_start", 18: "action_end"},
    "mHealth": {0 :"stand",1 :"sit",2 :"lie",3 :"walk",4 :"upstair",5 :"WaistBendForward",6 :"FrontalElevationArms",7 :"KneesBending", 8 :"Cycle",9 :"Jog",10: "Run",11: "Jump"}
}
repr_dim_dict = {"50salads":32, "HAPT":8, "mHealth": 32, "WISDM": 8} 
epoch_dict = {"50salads":100, "HAPT":50, "mHealth": 50, "WISDM": 10}
window_dict = {"50salads":50, "HAPT":100, "mHealth": 100, "WISDM": 50}


parser = argparse.ArgumentParser(description='cpd experiment')
parser.add_argument('--data', type=str, default='HAPT')
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--dim', type=int, default=-1)
parser.add_argument('--repr', type=str, default="TSCP2")
parser.add_argument('--window', type=int, default=50)
parser.add_argument('--slide', type=int, default=10)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--epoch', type=int, default=-1)
parser.add_argument('--seed', type=int, default=0, help="seed for each experiment")
parser.add_argument('--train', type=int, default=1, help="1 re-trains the representation from scratch")

parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--splr', type=int, default=100, help="sampling rate for representation visualization")
parser.add_argument('--lambd', type=int, default=80, help="hyperparam for cp metric")

parser.add_argument('--nnum', type=int, default=20, help="number of neighbors or non-neighboring timestamps")
parser.add_argument('--nrange', type=int, default=100, help="range of neighbors")

args = parser.parse_args()

device = "cuda:" + args.gpu
data_name = args.data
repr_name = args.repr
window_size = window_dict[data_name]
slide_size = args.slide
batch_size = args.batch
epochs = args.epoch
train = args.train

seed = args.seed

LR=args.lr
depth=args.depth
sampling_rate = args.splr
classlabel_dict = data_classlabel_dict[data_name]
lambd = args.lambd
nnum = args.nnum
nrange = args.nrange

input_dims = input_dims_dict[data_name]
parser.add_argument('--input_dim', type=int, default=input_dims)
args = parser.parse_args()
if repr_name == "TSCP2" and args.epoch==-1:
    args.epoch = epoch_dict[data_name]
elif repr_name == "TSCP2" and args.epoch!=-1:
    pass
else:
    args.epoch = 10
dim = args.dim
if dim == -1:
    args.dim = repr_dim_dict[data_name]
    dim = args.dim
else:
    pass
print(args)


train_labels = np.load(f"./datasets/{data_name}_y_long.npy")
abrupt_cp_long = np.load(f"./datasets/{data_name}_cp_long.npy") # 1 means gradual, 2 means abrupt

abrupt_CPs = np.where(abrupt_cp_long==2)[0]
all_CPs = np.where(train_labels[1:] != train_labels[:-1])[0]
gradual_CPs = set(all_CPs)
gradual_CPs.difference_update(set(abrupt_CPs))
gradual_CPs = list(gradual_CPs)
gradual_CPs.sort()
gradual_CPs = np.array(gradual_CPs)

boundary_labels, ts_list_per_abruptness = generate_gradual_bls(
    gradual_CPs.tolist(), 
    abrupt_CPs.tolist(),
    total_len=len(train_labels),
    ratio=0.1
)


if not os.path.exists('results/repr/'):
    os.makedirs('results/repr/')

if train:
    rm = ReprModel(args)
    test_repr_long = rm.fit()
    np.save(f"results/repr/repr_{data_name}_{repr_name}_{dim}_{seed}.npy", test_repr_long)
else:
    test_repr_long = np.load(f"results/repr/repr_{data_name}_{repr_name}_{dim}_{seed}.npy")

labels = train_labels[:len(test_repr_long)]
boundary_labels = boundary_labels[:len(test_repr_long)]
abrupt_cp_long = abrupt_cp_long[:len(test_repr_long)]

gradual_ts = ts_list_per_abruptness[0]
gradual_ts = np.array(gradual_ts)
gradual_ts = gradual_ts[gradual_ts<len(test_repr_long)]
abrupt_ts = ts_list_per_abruptness[1]
abrupt_ts = np.array(abrupt_ts)
abrupt_ts = abrupt_ts[abrupt_ts<len(test_repr_long)]


def evaluate(values):
    if len(gradual_ts) == 0:
        auc_g = 0
    else:
        auc_g = AUC(boundary_labels[gradual_ts], values[gradual_ts])
    auc_a = AUC(boundary_labels[abrupt_ts], values[abrupt_ts])
    auc_total = AUC(boundary_labels, values)
    threshold = best_f1_threshold(boundary_labels, values)
    loc = LOC(abrupt_cp_long, values>threshold)
    return auc_g, auc_a, auc_total, loc

def nearest_even_integer(n):
    n = int(n)
    if n % 2 == 0:
        return n
    else:
        return n+1

www = nearest_even_integer(mean_segment_length[data_name]/10)

curv, curv_r, curv_movavg = curvature_estimation(test_repr_long, www, device,20)
_, _, auc_total, loc = evaluate(curv)
print(auc_total, loc)