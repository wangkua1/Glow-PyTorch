import argparse
import os
import json
import shutil
import random
from itertools import islice

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from datasets import get_CIFAR10, get_SVHN, postprocess
from model import Glow
import torchvision.utils as vutils
from csv_logger import CSVLogger, plot_csv
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from inception_score import inception_score, run_fid

def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print('Using seed: {seed}'.format(seed=seed))

def cycle(loader):
    while True:
        for data in loader:
            yield data
            
def check_dataset(dataset, dataroot, augment, download):
    if dataset == 'cifar10':
        cifar10 = get_CIFAR10(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = cifar10
    if dataset == 'svhn':
        svhn = get_SVHN(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = svhn

    return input_size, num_classes, train_dataset, test_dataset


def compute_loss(nll, reduction='mean'):
    if reduction == 'mean':
        losses = {'nll': torch.mean(nll)}
    elif reduction == 'none':
        losses = {'nll': nll}

    losses['total_loss'] = losses['nll']

    return losses


def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction='mean'):
    if reduction == 'mean':
        losses = {'nll': torch.mean(nll)}
    elif reduction == 'none':
        losses = {'nll': nll}

    if multi_class:
        y_logits = torch.sigmoid(y_logits)
        loss_classes = F.binary_cross_entropy_with_logits(y_logits,
                                                          y,
                                                          reduction=reduction)
    else:
        loss_classes = F.cross_entropy(y_logits,
                                       torch.argmax(y, dim=1),
                                       reduction=reduction)

    losses['loss_classes'] = loss_classes
    losses['total_loss'] = losses['nll'] + y_weight * loss_classes

    return losses

def generate_from_noise(model, batch_size, device, clamp=False, guard_nans=True):
    _, c2, h, w  = model.prior_h.shape
    c = c2 // 2
    zshape = (batch_size, c, h, w)
    randz  = torch.randn(zshape).to(device)
    randz  = torch.autograd.Variable(randz, requires_grad=True)
    if clamp:
        randz = torch.clamp(randz,-5,5)
    images = model(z= randz, y_onehot=None, temperature=1, reverse=True) 
    if guard_nans:
        images[(images!=images)] = 0
    return images

def main(dataset, dataroot, download, augment, batch_size, eval_batch_size,
         epochs, saved_model, seed, hidden_channels, K, L, actnorm_scale,
         flow_permutation, flow_coupling, LU_decomposed, learn_top,
         y_condition, y_weight, max_grad_clip, max_grad_norm, lr,
         n_workers, cuda, n_init_batches, output_dir,
         saved_optimizer, warmup, fresh, db):

    device = 'cpu' if (not torch.cuda.is_available() or not cuda) else 'cuda:0'

    check_manual_seed(seed)

    ds = check_dataset(dataset, dataroot, augment, download)
    image_shape, num_classes, train_dataset, test_dataset = ds
    if db:
        train_dataset.data  = train_dataset.data[:1000]
        train_dataset.targets  = train_dataset.targets[:1000]
        test_dataset.data  = test_dataset.data[:1000]
        test_dataset.targets  = test_dataset.targets[:1000]
    # Note: unsupported for now
    multi_class = False

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=n_workers,
                                   drop_last=True)
    test_loader = data.DataLoader(test_dataset, batch_size=eval_batch_size,
                                  shuffle=False, num_workers=n_workers,
                                  drop_last=False)

    model = Glow(image_shape, hidden_channels, K, L, actnorm_scale,
                 flow_permutation, flow_coupling, LU_decomposed, num_classes,
                 learn_top, y_condition)

    model = model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=5e-5)

    lr_lambda = lambda epoch: lr * min(1., epoch / warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    iteration_fieldnames = ['global_iteration','train_bpd','test_bpd', 'fid']
    iteration_logger = CSVLogger(fieldnames=iteration_fieldnames,
                             filename=os.path.join(output_dir, 'iteration_log.csv'))
    

    def step(batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x = x.to(device)

        if y_condition:
            y = y.to(device)
            z, nll, y_logits = model(x, y)
            losses = compute_loss_y(nll, y_logits, y_weight, y, multi_class)
        else:
            z, nll, y_logits = model(x, None)
            losses = compute_loss(nll)

        losses['total_loss'].backward()

        if max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        return losses

    def eval_step(batch):
        model.eval()

        x, y = batch
        x = x.to(device)

        with torch.no_grad():
            if y_condition:
                y = y.to(device)
                z, nll, y_logits = model(x, y)
                losses = compute_loss_y(nll, y_logits, y_weight, y,
                                        multi_class, reduction='none')
            else:
                z, nll, y_logits = model(x, None)
                losses = compute_loss(nll)

        return losses


    # load pre-trained model if given
    if saved_model:
        model.load_state_dict(torch.load(saved_model))
        model.set_actnorm_init()

        if saved_optimizer:
            optimizer.load_state_dict(torch.load(saved_optimizer))

        file_name, ext = os.path.splitext(saved_model)
        resume_epoch = int(file_name.split('_')[-1])
        start_epoch = resume_epoch
    else:
        # Init
        model.train()
        init_batches = []
        init_targets = []

        with torch.no_grad():
            for batch, target in islice(train_loader, None,
                                        n_init_batches):
                init_batches.append(batch)
                init_targets.append(target)

            init_batches = torch.cat(init_batches).to(device)

            assert init_batches.shape[0] == n_init_batches * batch_size

            if y_condition:
                init_targets = torch.cat(init_targets).to(device)
            else:
                init_targets = None

            model(init_batches, init_targets)
        start_epoch = 1
    # 
    test_iter = cycle(test_loader)
    for epoch in range(start_epoch , epochs):
        scheduler.step()

        # ckpt
        torch.save(model.state_dict(), os.path.join(output_dir,  f'glow_model_{epoch}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir,  f'glow_optimizer_{epoch}.pth'))
        if epoch > 1 and os.path.exists(os.path.join(output_dir,  f'glow_model_{epoch-1}.pth')): # remove the previous ckpt to save space
            os.remove(os.path.join(output_dir,  f'glow_model_{epoch-1}.pth'))
            os.remove(os.path.join(output_dir,  f'glow_optimizer_{epoch-1}.pth'))
    
        # evaluate
        losses = defaultdict(list)
        pbar = tqdm(test_loader)
        for batch in pbar:
            cl = eval_step(batch)
            for k in cl:
                losses[k].append(cl[k].item())
            lstr = ', '.join([f"{k}: {np.mean(v):.3f}" for k, v in losses.items()])
            pbar.set_postfix_str(f"Eval -- {lstr}")
        losses = dict([(k, np.mean(v)) for k, v in losses.items()])
        test_bpd = losses['nll']
        # fid 
        with torch.no_grad():
            # Inception score
            N = 1000
            sample = torch.cat([generate_from_noise(model, eval_batch_size, device, clamp=False) for _ in range(200//eval_batch_size+1)],0 )[:N]
            sample = sample + .5
            
            x_real = torch.cat([test_iter.__next__()[0].to(device) for _ in range(200//eval_batch_size+1)],0 )[:N]
            x_real = x_real + .5
            if (sample!=sample).float().sum() > 0:
                myprint("Sample NaNs")
                raise
            fid =  run_fid(x_real.clamp_(0,1),sample.clamp_(0,1) )

        # vis
        samples = generate_from_noise(model, 64, device, clamp=False) 
        vutils.save_image(postprocess(samples), os.path.join(output_dir, f'samples_{epoch}.jpeg') , normalize=True, nrow=8) 

        # Train
        losses = defaultdict(list)
        pbar = tqdm(train_loader)
        for batch in pbar:
            cl = step(batch)
            for k in cl:
                losses[k].append(cl[k].item())
            lstr = ', '.join([f"{k}: {np.mean(v):.3f}" for k, v in losses.items()])
            pbar.set_postfix_str(f"Train -- {lstr}")
        losses = dict([(k, np.mean(v)) for k, v in losses.items()])
        train_bpd = losses['nll']
        

        # log 
        stats_dict = {
                'global_iteration': epoch ,
                'test_bpd': test_bpd,
                'train_bpd': train_bpd,
                'fid': fid
                }
        iteration_logger.writerow(stats_dict)
        plot_csv(iteration_logger.filename)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        default='cifar10', choices=['cifar10', 'svhn'],
                        help='Type of the dataset to be used.')

    parser.add_argument('--dataroot',
                        type=str, default='./',
                        help='path to dataset')

    parser.add_argument('--download', action='store_true',
                        help='downloads dataset')

    parser.add_argument('--no_augment', action='store_false',
                        dest='augment', help='Augment training data')

    parser.add_argument('--hidden_channels',
                        type=int, default=512,
                        help='Number of hidden channels')

    parser.add_argument('--K',
                        type=int, default=32,
                        help='Number of layers per block')

    parser.add_argument('--L',
                        type=int, default=3,
                        help='Number of blocks')

    parser.add_argument('--actnorm_scale',
                        type=float, default=1.0,
                        help='Act norm scale')

    parser.add_argument('--flow_permutation', type=str,
                        default='invconv', choices=['invconv', 'shuffle', 'reverse'],
                        help='Type of flow permutation')

    parser.add_argument('--flow_coupling', type=str,
                        default='affine', choices=['additive', 'affine'],
                        help='Type of flow coupling')

    parser.add_argument('--no_LU_decomposed', action='store_false',
                        dest='LU_decomposed',
                        help='Train with LU decomposed 1x1 convs')

    parser.add_argument('--learn_top', default=1, type=int)

    parser.add_argument('--y_condition', action='store_true',
                        help='Train using class condition')

    parser.add_argument('--y_weight',
                        type=float, default=0.01,
                        help='Weight for class condition loss')

    parser.add_argument('--max_grad_clip',
                        type=float, default=0,
                        help='Max gradient value (clip above - for off)')

    parser.add_argument('--max_grad_norm',
                        type=float, default=0,
                        help='Max norm of gradient (clip above - 0 for off)')

    parser.add_argument('--n_workers',
                        type=int, default=6,
                        help='number of data loading workers')

    parser.add_argument('--batch_size',
                        type=int, default=64,
                        help='batch size used during training')

    parser.add_argument('--eval_batch_size',
                        type=int, default=512,
                        help='batch size used during evaluation')

    parser.add_argument('--epochs',
                        type=int, default=250,
                        help='number of epochs to train for')

    parser.add_argument('--lr',
                        type=float, default=5e-4,
                        help='initial learning rate')

    parser.add_argument('--warmup',
                        type=float, default=5,
                        help='Warmup learning rate linearly per epoch')

    parser.add_argument('--n_init_batches',
                        type=int, default=8,
                        help='Number of batches to use for Act Norm initialisation')

    parser.add_argument('--no_cuda',
                        action='store_false',
                        dest='cuda',
                        help='disables cuda')

    parser.add_argument('--output_dir',
                        default='output/',
                        help='directory to output logs and model checkpoints')

    parser.add_argument('--fresh',
                        action='store_true',
                        help='Remove output directory before starting')

    parser.add_argument('--saved_model',
                        default='',
                        help='Path to model to load for continuing training')

    parser.add_argument('--saved_optimizer',
                        default='',
                        help='Path to optimizer to load for continuing training')

    parser.add_argument('--seed',
                        type=int, default=0,
                        help='manual seed')
    parser.add_argument('--db', action='store_true')
    
    args = parser.parse_args()
    kwargs = vars(args)

    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        if args.fresh:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        if (not os.path.isdir(args.output_dir)) or (len(os.listdir(args.output_dir)) > 0):
            raise FileExistsError("Please provide a path to a non-existing or empty directory. Alternatively, pass the --fresh flag.")

    with open(os.path.join(args.output_dir, 'hparams.json'), 'w') as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    main(**kwargs)
