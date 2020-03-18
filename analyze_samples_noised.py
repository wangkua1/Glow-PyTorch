
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
    # ipdb.set_trace()
    ydata =  torch.range(0,bpds.size(1)-1)[None].repeat(bpds.shape[0],1)
    plt.scatter(bpds.detach().cpu().numpy(), ydata.cpu().numpy(), c='b', alpha=.4, s=3)
    plt.scatter(clean_bpd.detach().cpu().numpy(), np.arange(clean_bpd.size(0)), c='r',marker='+', alpha=1, s=5)
    plt.plot(np.zeros(len(clean_bpd)), np.arange(len(clean_bpd)), label='0', c='k')

    #
    # plt.xscale('log')
    plt.legend()
    plt.grid(True, which='minor', axis='x')
    plt.xlim([-100,bpds.max().item()])
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


    
    x_test = torch.cat([test_iter.__next__()[0].to(device) for _ in range(4)], 0)
    x_train = torch.cat([train_iter.__next__()[0].to(device) for _ in range(4)], 0)
    

    assert kwargs['saved_model']
    print("Loading...")
    print(kwargs['saved_model'])
    sample_bpds = []
    test_bpds = []
    train_bpds = []
    idxs = range(0, 20000, 5000)
    for idx in tqdm(idxs):
        model =  torch.load(os.path.join(os.path.dirname(kwargs['saved_model']), f'ckpt_{idx}.pt'))
        with torch.no_grad():
            fake = generate_from_noise(model, 500)
            sample_bpd = torch.stack([model.forward(fake + 0.01*torch.randn_like(fake).to(device), None, return_details=True)[-1][0] * -1 for _ in range(30)])
            bpd = model.forward(fake, None, return_details=True)[-1][0] * -1 
            plot_bpds(sample_bpd, bpd,os.path.join(kwargs['output_dir'], f'sample_bpds_{idx}.png'))
            sample_bpds.append((sample_bpd, bpd))
        grid = make_grid((postprocess(fake.detach().cpu())[:30]), nrow=6).permute(1,2,0)
        plt.figure(figsize=(10,10))
        plt.imshow(grid)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(kwargs['output_dir'], f'sample_{idx}.png'), bbox_inches='tight', pad_inches=0)


        
        #
        with torch.no_grad():
            test_bpd = torch.stack([model.forward(x_test + 0.01*torch.randn_like(x_test).to(device), None, return_details=True)[-1][0] * -1 for _ in range(30)])
            bpd = model.forward(x_test, None, return_details=True)[-1][0] * -1 
            plot_bpds(test_bpd, bpd, os.path.join(kwargs['output_dir'], f'test_bpds_{idx}.png'))
            test_bpds.append((test_bpd, bpd))
            train_bpd = torch.stack([model.forward(x_train + 0.01*torch.randn_like(x_train).to(device), None, return_details=True)[-1][0] * -1 for _ in range(30)])
            bpd = model.forward(x_train, None, return_details=True)[-1][0] * -1 
            plot_bpds(train_bpd,bpd, os.path.join(kwargs['output_dir'], f'train_bpds_{idx}.png'))
            train_bpds.append((train_bpd, bpd))
    torch.save((sample_bpds, train_bpds, test_bpds), os.path.join(kwargs['output_dir'], f'bpds.pt') )
    # sample_bpds, train_bpds, test_bpds = torch.load(os.path.join(kwargs['output_dir'], f'bpds.pt') )
    plot_bpd_stats(sample_bpds, os.path.join(kwargs['output_dir'], f'stats_sample_bpds.png'))
    plot_bpd_stats(train_bpds, os.path.join(kwargs['output_dir'], f'stats_train_bpds.png'))
    plot_bpd_stats(test_bpds, os.path.join(kwargs['output_dir'], f'stats_test_bpds.png'))
            






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
