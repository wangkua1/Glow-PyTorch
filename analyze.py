import argparse
import os
import json

import torch
import torch.utils.data as data

from model import Glow
import matplotlib.pyplot as plt
import ipdb
from recon_mnist import run_recon_evolution
import utils
device = 'cpu' if (not torch.cuda.is_available()) else 'cuda:0'

import numpy as np
from anomaly import load_ood_data
from train import check_dataset, generate_from_noise
import math
from collections import OrderedDict
from datasets import preprocess
from torchvision import transforms

c, h, w = 3,32,32
n_bins = 2**8
chw = c * h * w
bpd_correction = -math.log(n_bins) / (math.log(2.))

def compute_percent_nans_infs(x):
    x = x.view(x.size(0), -1)
    n, d = x.shape
    nx_nans = ((x!=x).sum(-1) > 0 ).sum()
    n_nans = (x!=x).sum()
    nx_infs = ((x==np.inf).sum(-1) > 0).sum()
    n_infs = (x==np.inf).sum()
    return (n_nans + n_infs).float() / float(n*d), (nx_nans + nx_infs).float() / float(n)


def compute_jac_cn(x, model):
    dic = utils.computeSVDjacobian(x, model, compute_inverse=False)
    D_for, jac = dic['D_for'], dic['jac_for']
    cn = float(D_for.max()/ D_for.min())
    return cn, jac

def run_analysis(x, model, recon_path):
    p_pxs, p_ims = compute_percent_nans_infs(x)
    
    # Note: CN is computed only for the 1st sample
    cn, jac = compute_jac_cn(x, model)
    _, numerical_logdet = np.linalg.slogdet(jac)
    
    with torch.no_grad():
        _, bpd, _, (_, analytic_logdet) = model.forward(x, None, return_details=True, correction=False)
        # Subtract the conditional gaussian likelihood from the split layers
        analytic_logdet = analytic_logdet - torch.stack([split._last_logdet for  split  in model.flow.splits]).sum(0)
        # The above forward pass was run w/o correction
        data_bpd = bpd.mean().item() - bpd_correction 

    with torch.no_grad():
        data_pad = run_recon_evolution(model, 
                                        x, 
                                        recon_path)
    return p_pxs.item(), p_ims.item(), cn, np.abs(numerical_logdet-analytic_logdet[0].item()), data_bpd, data_pad.item()

def one_to_three_channels(x):
    if x.shape[0] == 1:
        x = x.repeat(3,1,1)
    return x   

def main(dataset, dataroot, download, augment, n_workers, eval_batch_size, output_dir,db, glow_path,ckpt_name):

    
    (image_shape, num_classes, train_dataset, test_dataset) = check_dataset(dataset, dataroot, augment, download)

    test_loader = data.DataLoader(test_dataset, batch_size=eval_batch_size,
                                  shuffle=False, num_workers=n_workers,
                                  drop_last=False)

    x = test_loader.__iter__().__next__()[0].to(device)

    # OOD data
    ood_distributions = ['gaussian', 'rademacher', 'texture3', 'svhn','tinyimagenet','lsun']
    tr = transforms.Compose([])
    tr.transforms.append(transforms.ToPILImage()) 
    tr.transforms.append(transforms.Resize((32,32)))
    tr.transforms.append(transforms.ToTensor())
    tr.transforms.append(one_to_three_channels)
    tr.transforms.append(preprocess)
    ood_tensors = [(out_name, torch.stack([tr(x) for x in load_ood_data({
                                  'name': out_name,
                                  'ood_scale': 1,
                                  'n_anom': eval_batch_size,
                                })]).to(device)
                        ) for out_name in ood_distributions]

    model = torch.load(glow_path)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        samples = generate_from_noise(model, eval_batch_size,clamp=False, guard_nans=False)
    stats = OrderedDict()
    for name, x in [('data',x), ('samples',samples)] + ood_tensors:
        p_pxs, p_ims, cn, dlogdet, bpd, pad = run_analysis(x, model, os.path.join(output_dir, f'recon_{ckpt_name}_{name}.jpeg'))
        
        stats[f"{name}-percent-pixels-nans"] =  p_pxs
        stats[f"{name}-percent-imgs-nans"] =  p_ims
        stats[f"{name}-cn"] =  cn
        stats[f"{name}-dlogdet"] =  dlogdet
        stats[f"{name}-bpd"] =  bpd
        stats[f"{name}-recon-err"] =  pad
        
        with open(os.path.join(output_dir, f'results_{ckpt_name}.json'), 'w') as fp:
            json.dump(stats, fp, indent=4)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        default='cifar10', choices=['cifar10', 'svhn', 'mnist'],
                        help='Type of the dataset to be used.')
    parser.add_argument('--dataroot',
                        type=str, default='/scratch/gobi2/wangkuan/data',
                        help='path to dataset')
    parser.add_argument('--download', default=True)
    parser.add_argument('--no_augment', action='store_false',
                        dest='augment', help='Augment training data')
    parser.add_argument('--n_workers',
                        type=int, default=6,
                        help='number of data loading workers')
    parser.add_argument('--eval_batch_size',
                        type=int, default=512,
                        help='batch size used during evaluation')
    parser.add_argument('--db', type=int, default=0)
    parser.add_argument('--glow_path', type=str, default='')

    args = parser.parse_args()
    kwargs = vars(args)

    # Create output_dir 
    base_dir = os.path.dirname(args.glow_path)
    args.output_dir = os.path.join(base_dir, 'analyze')
    args.ckpt_name = os.path.basename(args.glow_path).split('.')[0]


    makedirs(args.dataroot)
    makedirs(args.output_dir)
    
    with open(os.path.join(args.output_dir, f'hparams_{args.ckpt_name}.json'), 'w') as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    log_file = os.path.join(args.output_dir, f'log_{args.ckpt_name}.txt')
    log = open(log_file, 'w')
    _print = print
    def print(*content):
        _print(*content)
        _print(*content, file=log)
        log.flush()

    main(**kwargs)
    log.close()
