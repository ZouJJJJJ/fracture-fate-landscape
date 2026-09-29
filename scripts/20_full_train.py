# -*- coding: utf-8 -*-
"""WP1 full RUOT training on GSE185940 (Tam four-gate daily E11.5-E14.5), CPU.
Faithful 3-stage schedule (score pretrain -> velocity/growth pretrain -> total RUOT),
with periodic checkpointing and loss logging. Counts via env vars."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os, types, time, csv, random, gc
BASE = ''+ROOT+''
# (third-party dependencies are installed via requirements.txt)
sys.path.append(os.path.join(BASE, 'vendor/DiffusionOT'))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, torch.nn as nn
from functools import partial
import utility as U
from utility import RUOT, initialize_weights

N_SCORE = int(os.environ.get('N_SCORE', 80))
N_PRE = int(os.environ.get('N_PRE', 50))
N_TOTAL = int(os.environ.get('N_TOTAL', 500))
NUM_SAMPLES = int(os.environ.get('NUM_SAMPLES', 128))

# speed: cap anchor-gaussian sample size in kernel density (default 500 -> 200)
_orig_dens = U.MultimodalGaussian_density_sample
def _fast_dens(x, time_all, time_pt, data_train, sigma, device, sample_size=200):
    return _orig_dens(x, time_all, time_pt, data_train, sigma, device, sample_size=sample_size)
U.MultimodalGaussian_density_sample = _fast_dens

torch.manual_seed(1); random.seed(1); np.random.seed(1)
device = torch.device('cpu')
args = types.SimpleNamespace(dataset='gse185940_full', niters=N_TOTAL, lr=3e-3,
                             num_samples=NUM_SAMPLES, hidden_dim=16, n_hiddens=3,
                             activation='Tanh', d=0.001, seed=1)
save_dir = os.path.join(BASE, 'wp1_out'); os.makedirs(save_dir, exist_ok=True)

d = np.load(os.path.join(BASE, 'data/gse185940/gse185940_latent_full.npz'), allow_pickle=True)
latent, tl = d['pca_scaled'], d['time_label'].astype(str)
data_train = []
for k in range(4):
    idx = np.where(tl == str(k + 1))[0]
    data_train.append(torch.from_numpy(latent[idx]).float().to(device))
print('clouds:', [tuple(c.shape) for c in data_train], flush=True)
integral_time = [0., 1., 2., 3.]
train_time = list(range(len(data_train)))
options = {'method': 'Dopri5', 'h': None, 'rtol': 1e-3, 'atol': 1e-5,
           'print_neval': False, 'neval_max': 1000000, 'safety': None}
mse = nn.MSELoss()

func = RUOT(in_out_dim=data_train[0].shape[1], hidden_dim=args.hidden_dim,
            n_hiddens=args.n_hiddens, activation=args.activation, d=args.d).to(device)
func.apply(initialize_weights)

losslog = os.path.join(save_dir, 'loss_history.csv')
lf = open(losslog, 'w', newline=''); lcw = csv.writer(lf); lcw.writerow(['stage', 'iter', 'loss', 'sec'])
def log(stage, it, l, sec):
    lcw.writerow([stage, it, round(float(l), 6), round(sec, 3)]); lf.flush()

def setreq(layers_on, d_on=False):
    func.d.requires_grad = d_on
    for net in [func.hyper_net1, func.hyper_net2, func.hyper_net3]:
        for p in net.parameters(): p.requires_grad = False
    for net in layers_on:
        for p in net.parameters(): p.requires_grad = True

# ---- Stage 1: score pretrain (hyper_net3) ----
setreq([func.hyper_net3])
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, func.parameters()), lr=args.lr, weight_decay=0.01)
for it in range(1, N_SCORE + 1):
    t0 = time.time(); opt.zero_grad()
    l = U.pre_train_score(mse, func, args, data_train, train_time, integral_time, 1, device, it)
    l.backward(); opt.step(); log('score', it, l, time.time() - t0)
    if it % 10 == 0 or it <= 2: print('score', it, round(l.item(), 4), flush=True)
torch.save({'func_state_dict': func.state_dict()}, os.path.join(save_dir, 'ckpt_score.pth'))

# ---- Stage 2: velocity/growth pretrain (hyper_net1, hyper_net2) ----
setreq([func.hyper_net1, func.hyper_net2])
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, func.parameters()), lr=args.lr, weight_decay=0.01)
lr_adj = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[N_TOTAL - 400, N_TOTAL - 200], gamma=0.5)
for it in range(1, N_PRE + 1):
    t0 = time.time(); opt.zero_grad()
    l, sigma, l3, l4 = U.pre_train_model(mse, func, args, data_train, train_time,
                                         integral_time, 1, options, device, it)
    l.backward(); opt.step(); lr_adj.step(); log('pre', it, l, time.time() - t0)
    if it % 10 == 0 or it <= 2: print('pre', it, round(l.item(), 4), flush=True)
torch.save({'func_state_dict': func.state_dict()}, os.path.join(save_dir, 'ckpt_pre.pth'))

# ---- Stage 3: total RUOT (all nets) ----
setreq([func.hyper_net1, func.hyper_net2, func.hyper_net3])
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, func.parameters()), lr=args.lr, weight_decay=0.01)
lr_adj = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[N_TOTAL - 400, N_TOTAL - 200], gamma=0.5)
tstart = time.time()
for it in range(1, N_TOTAL + 1):
    t0 = time.time(); opt.zero_grad()
    l, l1, sigma, v1, v2, v3, v4 = U.train_model(mse, func, args, data_train, train_time,
                                                integral_time, 0.5, options, device, it)
    l.backward(); opt.step(); lr_adj.step(); log('total', it, l, time.time() - t0)
    if it % 10 == 0 or it <= 3:
        print('total', it, round(l.item(), 4), 'avgsec/it', round((time.time()-tstart)/it,2), flush=True)
    if it % 50 == 0:
        torch.save({'func_state_dict': func.state_dict(), 'iter': it},
                   os.path.join(save_dir, 'ckpt_total_last.pth'))
torch.save({'func_state_dict': func.state_dict(), 'iter': N_TOTAL,
            'integral_time': integral_time}, os.path.join(save_dir, 'ckpt_final.pth'))
lf.close()
print('WP1 TRAINING DONE', N_TOTAL, 'total iters in', round((time.time()-tstart)/60, 1), 'min', flush=True)
