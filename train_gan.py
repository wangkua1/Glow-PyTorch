"""
python train.py      
--fresh      
--dataset mnist      
--L 3      
--K 32     
--hidden_channels 128      
--batch_size 64     
--lr 1e-5      
--disc_lr 1e-4     
--flow_permutation reverse       
--flow_coupling affine      
--gan      
--output_dir /scratch/gobi2/wangkuan/glow/db-gan2
"""
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

def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    myprint('Using seed: {seed}'.format(seed=seed))

def gradient_penalty(x, y, f):
    """From https://github.com/LynnHo/Pytorch-WGAN-GP-DRAGAN-Celeba/blob/master/train_celeba_wgan_gp.py
    """
    # Interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape, device='cuda:0', requires_grad=True)
    z = x + alpha * (y - x)

    # Gradient penalty
    o = f(z)
    g = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size(), device=x.device), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()
    return gp


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


def main(dataset, dataroot, download, augment, batch_size, eval_batch_size,
         epochs, saved_model, seed, hidden_channels, K, L, actnorm_scale,
         flow_permutation, flow_coupling, LU_decomposed, learn_top,
         y_condition, y_weight, max_grad_clip, max_grad_norm, lr,
         n_workers, cuda, n_init_batches, warmup_steps, output_dir,
         saved_optimizer, warmup, fresh,logittransform, gan, disc_lr):

    device = 'cpu' if (not torch.cuda.is_available() or not cuda) else 'cuda:0'

    check_manual_seed(seed)

    ds = check_dataset(dataset, dataroot, augment, download)
    image_shape, num_classes, train_dataset, test_dataset = ds

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
                 learn_top, y_condition,logittransform)

    model = model.to(device)
    
    if gan:
        # Debug
        model = mine.Generator(32, 1).to(device)


        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(.5, .99), weight_decay=0)
        discriminator = mine.Discriminator(image_shape[-1])
        discriminator = discriminator.to(device)
        D_optimizer = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=disc_lr, betas=(.5, .99), weight_decay=0)
    else:
        optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=5e-5)



    # lr_lambda = lambda epoch: lr * min(1., epoch+1 / warmup)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    i = 0
    def step(engine, batch):
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

    def gan_step(engine, batch):
        assert not y_condition
        if 'iter_ind' in dir(engine):
            engine.iter_ind += 1
        else:
            engine.iter_ind = -1
        losses = {}
        model.train()
        discriminator.train()
        

        x, y = batch
        x = x.to(device)

        
        # def generate_from_noise(batch_size):
        #     _, c2, h, w  = model.prior_h.shape
        #     c = c2 // 2
        #     zshape = (batch_size, c, h, w)
        #     randz  = torch.autograd.Variable(torch.randn(zshape), requires_grad=True).to(device)
        #     images = model(z= randz, y_onehot=None, temperature=1, reverse=True,batch_size=batch_size)   
        #     return images

        def generate_from_noise(batch_size):

            zshape = (batch_size, 32, 1,1)
            randz  = torch.randn(zshape).to(device)
            images = model(randz)   
            return images / 2

        def run_noised_disc(discriminator, x):
            x = uniform_binning_correction(x)[0]
            return discriminator(x)
        
        # Train Disc
        fake = generate_from_noise(x.size(0))

        D_real_scores = run_noised_disc(discriminator, x.detach())
        D_fake_scores = run_noised_disc(discriminator, fake.detach())

        ones_target = torch.ones((x.size(0), 1), device=x.device)
        zeros_target = torch.zeros((x.size(0), 1), device=x.device)

        # D_real_accuracy = torch.sum(torch.round(F.sigmoid(D_real_scores)) == ones_target).float() / ones_target.size(0)
        # D_fake_accuracy = torch.sum(torch.round(F.sigmoid(D_fake_scores)) == zeros_target).float() / zeros_target.size(0)

        D_real_loss = F.binary_cross_entropy_with_logits(D_real_scores, ones_target)
        D_fake_loss = F.binary_cross_entropy_with_logits(D_fake_scores, zeros_target)

        D_loss = (D_real_loss + D_fake_loss) / 2
        gp = gradient_penalty(x.detach(), fake.detach(), lambda _x: run_noised_disc(discriminator, _x))
        D_loss_plus_gp = D_loss  + 10*gp
        D_optimizer.zero_grad()
        D_loss_plus_gp.backward()
        D_optimizer.step()


        # Train generator
        fake = generate_from_noise(x.size(0))
        G_loss = F.binary_cross_entropy_with_logits(run_noised_disc(discriminator, fake), torch.ones((x.size(0), 1), device=x.device))
        losses['total_loss'] = G_loss

        # G-step
        optimizer.zero_grad()
        losses['total_loss'].backward()
        params = list(model.parameters())
        gnorm = [p.grad.norm() for p in params]
        optimizer.step()
        # if max_grad_clip > 0:
        #     torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
        # if max_grad_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


        if engine.iter_ind  % 50==0:
            grid = make_grid((postprocess(fake.detach().cpu())[:30]), nrow=6).permute(1,2,0)
            plt.figure(figsize=(10,10))
            plt.imshow(grid)
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'sample_{engine.iter_ind}.png'))

            grid = make_grid((postprocess(uniform_binning_correction(x)[0].cpu())[:30]), nrow=6).permute(1,2,0)
            plt.figure(figsize=(10,10))
            plt.imshow(grid)
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'data_{engine.iter_ind}.png'))

        return losses


    def eval_step(engine, batch):
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
                losses = compute_loss(nll, reduction='none')

        return losses
    if gan:
        trainer = Engine(gan_step)
    else:
        trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(output_dir, 'glow', save_interval=1,
                                         n_saved=2, require_empty=False)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler,
                              {'model': model, 'optimizer': optimizer})

    monitoring_metrics = ['total_loss']
    RunningAverage(output_transform=lambda x: x['total_loss']).attach(trainer, 'total_loss')

    evaluator = Engine(eval_step)

    # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
    Loss(lambda x, y: torch.mean(x), output_transform=lambda x: (x['total_loss'], torch.empty(x['total_loss'].shape[0]))).attach(evaluator, 'total_loss')

    if y_condition:
        monitoring_metrics.extend(['nll'])
        RunningAverage(output_transform=lambda x: x['nll']).attach(trainer, 'nll')

        # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
        Loss(lambda x, y: torch.mean(x), output_transform=lambda x: (x['nll'], torch.empty(x['nll'].shape[0]))).attach(evaluator, 'nll')

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    # load pre-trained model if given
    if saved_model:
        model.load_state_dict(torch.load(saved_model))
        model.set_actnorm_init()

        if saved_optimizer:
            optimizer.load_state_dict(torch.load(saved_optimizer))

        file_name, ext = os.path.splitext(saved_model)
        resume_epoch = int(file_name.split('_')[-1])

        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.epoch = resume_epoch
            engine.state.iteration = resume_epoch * len(engine.state.dataloader)

    # @trainer.on(Events.STARTED)
    # def init(engine):
    #     model.train()

    #     init_batches = []
    #     init_targets = []

    #     with torch.no_grad():
    #         for batch, target in islice(train_loader, None,
    #                                     n_init_batches):
    #             init_batches.append(batch)
    #             init_targets.append(target)

    #         init_batches = torch.cat(init_batches).to(device)

    #         assert init_batches.shape[0] == n_init_batches * batch_size

    #         if y_condition:
    #             init_targets = torch.cat(init_targets).to(device)
    #         else:
    #             init_targets = None

    #         model(init_batches, init_targets)

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def evaluate(engine):
    #     evaluator.run(test_loader)

    #     # scheduler.step()
    #     metrics = evaluator.state.metrics

    #     losses = ', '.join([f"{key}: {value:.2f}" for key, value in metrics.items()])

    #     myprint(f'Validation Results - Epoch: {engine.state.epoch} {losses}')

    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message(f'Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]')
        timer.reset()

    trainer.run(train_loader, epochs)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if __name__ == '__main__':
    """ 
    python train.py \
    --output_dir /scratch/gobi2/wangkuan/glow/cifar10

    python train.py --no_learn_top --dataset mnist --L 2 \
    --hidden_channels 128 --lr 1e-3 \
    --flow_permutation reverse \
    --flow_coupling additive \
    --output_dir /scratch/gobi2/wangkuan/glow/db --fresh
    """
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

    parser.add_argument('--no_learn_top', action='store_false',
                        help='Do not train top layer (prior)', dest='learn_top')

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
                        type=float, default=1,
                        help='Warmup learning rate linearly per epoch')

    parser.add_argument('--warmup_steps',
                        type=int, default=4000,
                        help='Number of warmup steps for lr initialisation')

    parser.add_argument('--n_init_batches',
                        type=int, default=8,
                        help='Number of batches to use for Act Norm initialisation')

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

    parser.add_argument('--saved_optimizer',
                        default='',
                        help='Path to optimizer to load for continuing training')

    parser.add_argument('--seed',
                        type=int, default=0,
                        help='manual seed')
    parser.add_argument('--logittransform',
                        action='store_true')
    parser.add_argument('--gan',
                        action='store_true')
    parser.add_argument('--disc_lr',
                        type=float, default=1e-5)

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

    with open(os.path.join(args.output_dir, 'hparams.json'), 'w') as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    log_file = os.path.join(args.output_dir, 'log.txt')
    log = open(log_file, 'w')

    def myprint(*content):
        print(*content)
        print(*content, file=log)
        log.flush()

    main(**kwargs)
    log.close()
