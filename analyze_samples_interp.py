
import argparse
import os
import json
import shutil
import random
from itertools import islice
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss
from torchvision.utils import make_grid

from datasets import get_CIFAR10, get_SVHN, get_MNIST, postprocess
from model import Glow
import mine
import matplotlib.pyplot as plt
import ipdb
from utils import uniform_binning_correction
from recon_mnist import run_recon_evolution
from inception_score import inception_score, run_fid
from csv_logger import CSVLogger, plot_csv
import plot_utils
import utils
import numpy as np
from tqdm import tqdm
device = 'cpu' if (not torch.cuda.is_available()) else 'cuda:0'

def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    myprint('Using seed: {seed}'.format(seed=seed))


def check_dataset(dataset, dataroot, augment, download):
    if dataset == 'cifar10':
        cifar10 = get_CIFAR10(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = cifar10
    if dataset == 'svhn':
        svhn = get_SVHN(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = svhn

    if dataset == 'mnist':
        mnist = get_MNIST(False, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = mnist

    return input_size, num_classes, train_dataset, test_dataset




def cycle(loader):
    while True:
        for data in loader:
            yield data

def generate_from_noise(model, batch_size,clamp=False, guard_nans=True):
    _, c2, h, w  = model.prior_h.shape
    c = c2 // 2
    zshape = (batch_size, c, h, w)
    randz  = torch.randn(zshape).to(device)
    randz  = torch.autograd.Variable(randz, requires_grad=True)
    if clamp:
        randz = torch.clamp(randz,-5,5)
    images = model(z= randz, y_onehot=None, temperature=1, reverse=True,batch_size=0) 
    if guard_nans:
        images[(images!=images)] = 0
    return images

def plot_bpds(bpds, clean_bpd, fpath):
    # Replace NaNs
    bpds[(bpds!=bpds)] = -100
    # Replace Infs
    bpds[bpds == float('inf')] = -100
    plt.figure(figsize=(10,10))
    #
    plt.scatter(bpds.detach().cpu().numpy(), np.arange(bpds.size(0)), c='b', alpha=.4, s=3, label='perturbed')
    plt.scatter(clean_bpd.detach().cpu().numpy(), np.arange(clean_bpd.size(0)), c='r',marker='+', alpha=1, s=5, label='original')
    plt.plot(np.zeros(len(clean_bpd)), np.arange(len(clean_bpd)), label='0', c='k')

    #
    plt.legend()
    plt.grid(True, which='minor', axis='x')
    plt.xscale('log')
    plt.xlim([1e-2, 1e20])
    # plt.xlim([-100,bpds.max().item()])
    plt.tight_layout()
    plt.savefig(fpath, bbox_inches='tight', pad_inches=0)

def plot_bpd_stats(bpds, fpath):
    MAX = 1e38
    fig,  axs = plt.subplots(2,2,figsize=(10,7))
    # 
    xdata =  torch.range(0,len(bpds)-1).cpu()
    stats = []
    for (bpd, clean_bpd) in bpds:
        # Replace NaNs
        bpd[(bpd!=bpd)] = MAX
        # Replace Infs
        bpd[bpd == float('inf')] = MAX
        bpd[bpd < 0] = MAX
        # bpd[bpd > MAX] = MAX
        stats.append((
                bpd.max(0)[0].cpu(),
                bpd.min(0)[0].cpu(),
                bpd.mean(0).cpu(),
                clean_bpd.cpu()
            ))
    # ipdb.set_trace()
    for ax in [axs[0][0],  axs[0][1],  axs[1][0],  axs[1][1]]:
        ax.set_yscale('log')
        ax.set_ylim([1e-3,1e50])

    # ipdb.set_trace()
    axs[0][0].scatter(xdata[:,None].repeat(1,len(stats[0][0])).numpy(), 
                torch.stack([item[0] for item in stats]).numpy(), 
                c='b', alpha=.4, s=3)
    axs[0][0].set_title('max')
    axs[0][1].scatter(xdata[:,None].repeat(1,len(stats[0][0])).numpy(), 
                torch.stack([item[1] for item in stats]).numpy(), 
                c='b', alpha=.4, s=3)
    axs[0][1].set_title('min')
    axs[1][0].scatter(xdata[:,None].repeat(1,len(stats[0][0])).numpy(), 
                torch.stack([item[2] for item in stats]).numpy(), 
                c='b', alpha=.4, s=3)
    axs[1][0].set_title('mean')
    axs[1][1].scatter(xdata[:,None].repeat(1,len(stats[0][0])).numpy(), 
                torch.stack([item[3] for item in stats]).numpy(), 
                c='b', alpha=.4, s=3)
    axs[1][1].set_title('clean')

    #
    # plt.xscale('log')
    # plt.legend()

    plt.tight_layout()
    plt.savefig(fpath, bbox_inches='tight', pad_inches=0)


def l2_project(x, x_0, t=.1):
    d = x - x_0
    n = torch.pow(torch.pow(d , 2).sum(-1, keepdim=True),.5)
    div = n / t
    div.clamp_(min=1)
    return x_0 + d / div


def pgd(x, f_loss, step_size, n_steps, f_project):
    x_0 = x.clone()
    x.requires_grad_()
    for n in range(n_steps):
        g = torch.autograd.grad(f_loss(x).sum(), [x], create_graph=False)[0]
        g = g / (torch.norm(g)+1e-10)
        # if (g!=g).sum() > 0:
        #     ipdb.set_trace()
        x = x - step_size * g
        if f_project is not None:
            x = f_project(x, x_0)
    return x

def plot_samples(x, fpath):
    grid = make_grid((postprocess(x.detach().cpu())[:30]), nrow=6).permute(1,2,0)
    plt.figure(figsize=(5,5))
    plt.imshow(grid)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fpath, bbox_inches='tight', pad_inches=0)

def main(kwargs):

    
    check_manual_seed(kwargs['seed'])

    ds = check_dataset(kwargs['dataset'], kwargs['dataroot'], False, kwargs['download'])
    image_shape, num_classes, train_dataset, test_dataset = ds

    # Note: unsupported for now
    multi_class = False

    train_loader = data.DataLoader(train_dataset, batch_size=kwargs['batch_size'],
                                   shuffle=True, num_workers=kwargs['n_workers'],
                                   drop_last=True)
    test_loader = data.DataLoader(test_dataset, batch_size=kwargs['eval_batch_size'],
                                  shuffle=False, num_workers=kwargs['n_workers'],
                                  drop_last=False)
    test_iter = cycle(test_loader)
    train_iter = cycle(train_loader)


    
    x_test = torch.cat([test_iter.__next__()[0].to(device) for _ in range(1)], 0)
    x_train = torch.cat([train_iter.__next__()[0].to(device) for _ in range(1)], 0)
    

    if kwargs['pgd_f_project'] == 'l2':
        f_project = lambda x, x_0: l2_project(x, x_0, t=kwargs['pgd_l2_t'])
    elif kwargs['pgd_f_project'] == 'none':
        f_project = None
    else:
        raise
    run_pgd = lambda x, f_loss: pgd(x, 
                                     f_loss=f_loss, 
                                     step_size=kwargs['pgd_step_size'], 
                                     n_steps=kwargs['pgd_n_steps'], 
                                     f_project=f_project)

    assert kwargs['saved_model']
    print("Loading...")
    print(kwargs['saved_model'])
    sample_bpds = []
    test_bpds = []
    train_bpds = []
    idxs = range(0, 20000, 5000)
    for idx in tqdm(idxs):
        model =  torch.load(os.path.join(os.path.dirname(kwargs['saved_model']), f'ckpt_{idx}.pt'))
        model.eval()
        # ipdb.set_trace()

        # 1 sample
        x = x_train[:1].repeat(10,1,1,1).clone()
        z = model.forward(x, None, return_details=True, correction=False)[0]
         
        n_grid = 9
        samples = [x]
        for n in range(n_grid):
            curr_z = z * (1-float(n)/n_grid)
            s = model(y_onehot=None, temperature=1, z=curr_z, reverse=True, use_last_split=True)
            samples.append(s)
        samples = torch.cat(samples, 0)
        grid = make_grid((postprocess(samples.detach().cpu())), nrow=10).permute(1,2,0)
        plt.figure(figsize=(5,5))
        plt.imshow(grid)
        plt.axis('off')
        plt.tight_layout()
        fpath = os.path.join(kwargs['output_dir'], f'linear_last_{idx}.png')
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0)

        n_grid = 9
        z = model.forward(x, None, return_details=True, correction=False)[0]
        samples = [x]
        for n in range(n_grid):
            curr_z = z * (1-float(n)/n_grid)
            s = model(y_onehot=None, temperature=1, z=curr_z, reverse=True, use_last_split=False)
            samples.append(s)
        samples = torch.cat(samples, 0)
        grid = make_grid((postprocess(samples.detach().cpu())), nrow=10).permute(1,2,0)
        plt.figure(figsize=(5,5))
        plt.imshow(grid)
        plt.axis('off')
        plt.tight_layout()
        fpath = os.path.join(kwargs['output_dir'], f'linear_no_last_{idx}.png')
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0)

        n_grid = 9
        z = model.forward(x, None, return_details=True, correction=False)[0]
        s = model(y_onehot=None, temperature=1, z=z, reverse=True, use_last_split=True)
        samples = [s]
        for n in range(n_grid):
            m = np.logspace(0, np.log10(1 / z.view(10, -1).pow(2).sum(-1).pow(.5)[0].item()), n_grid)[n]
            curr_z = z * m
            s = model(y_onehot=None, temperature=1, z=curr_z, reverse=True, use_last_split=True)
            samples.append(s.clone())
        samples = torch.cat(samples, 0)
        grid = make_grid((postprocess(samples.detach().cpu())), nrow=10).permute(1,2,0)
        plt.figure(figsize=(5,5))
        plt.imshow(grid)
        plt.axis('off')
        plt.tight_layout()
        fpath = os.path.join(kwargs['output_dir'], f'log_last_{idx}.png')
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0)


        z = model.forward(x, None, return_details=True, correction=False)[0]
        samples = [x]
        for n in range(n_grid):
            curr_z = z * np.logspace(0, np.log10(1 / z.view(10, -1).pow(2).sum(-1).pow(.5)[0].item()), n_grid)[n]
            s = model(y_onehot=None, temperature=1, z=curr_z, reverse=True, use_last_split=False)
            samples.append(s)
        samples = torch.cat(samples, 0)
        grid = make_grid((postprocess(samples.detach().cpu())), nrow=10).permute(1,2,0)
        plt.figure(figsize=(5,5))
        plt.imshow(grid)
        plt.axis('off')
        plt.tight_layout()
        fpath = os.path.join(kwargs['output_dir'], f'log_no_last_{idx}.png')
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0)


        # random recons
        n_grid = 9
        z = model.forward(x, None, return_details=True, correction=False)[0]
        
        samples = [x]
        for n in range(n_grid):
            s = model(y_onehot=None, temperature=1, z=z, reverse=True, use_last_split=False)
            samples.append(s)
        samples = torch.cat(samples, 0)
        grid = make_grid((postprocess(samples.detach().cpu())), nrow=10).permute(1,2,0)
        plt.figure(figsize=(5,5))
        plt.imshow(grid)
        plt.axis('off')
        plt.tight_layout()
        fpath = os.path.join(kwargs['output_dir'], f'random_recons_{idx}.png')
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0)



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
    parser.add_argument('--n_workers',
                        type=int, default=6,
                        help='number of data loading workers')

    parser.add_argument('--batch_size',
                        type=int, default=500,
                        help='batch size used during training')

    parser.add_argument('--eval_batch_size',
                        type=int, default=500,
                        help='batch size used during evaluation')

    
    parser.add_argument('--no_cuda',
                        action='store_false',
                        dest='cuda',
                        help='disables cuda')

    parser.add_argument('--output_dir',
                        default='/scratch/gobi2/wangkuan/glow/',
                        help='directory to output logs and model checkpoints')

    parser.add_argument('--fresh',
                        action='store_true',
                        help='Remove output directory before starting')

    parser.add_argument('--saved_model',
                        default='',
                        help='Path to model to load for continuing training')

    
    parser.add_argument('--seed',
                        type=int, default=0,
                        help='manual seed')

    parser.add_argument('--pgd_step_size', type=float, default=0.01)
    parser.add_argument('--pgd_n_steps', type=int, default=10)
    parser.add_argument('--pgd_f_project', type=str, default='l2')
    parser.add_argument('--pgd_l2_t', type=float, default=0.01)
    args = parser.parse_args()
    kwargs = vars(args)

    makedirs(args.dataroot)

    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        if args.fresh:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        if (not os.path.isdir(args.output_dir)) or (len(os.listdir(args.output_dir)) > 0):
            raise FileExistsError("Please provide a path to a non-existing or empty directory. Alternatively, pass the --fresh flag.")
    os.makedirs(os.path.join(args.output_dir, '_recon'))

    with open(os.path.join(args.output_dir, 'hparams.json'), 'w') as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    log_file = os.path.join(args.output_dir, 'log.txt')
    log = open(log_file, 'w')

    def myprint(*content):
        print(*content)
        print(*content, file=log)
        log.flush()

    main(kwargs)
    log.close()
