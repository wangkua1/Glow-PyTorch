import json

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from datasets import get_CIFAR10, get_SVHN, get_MNIST, postprocess
from model import Glow
import ipdb
import os
device = torch.device("cuda")

output_folder = '/scratch/gobi2/wangkuan/glow/mnist-1x1-affine-512-1e-2'
model_name = 'glow_model_250.pth'

with open(os.path.join(output_folder,'hparams.json')) as json_file:  
    hparams = json.load(json_file)
  
image_shape, num_classes, _, test_mnist = get_MNIST(False, hparams['dataroot'], hparams['download'])


model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'], False if 'logittransform' not in hparams else hparams['logittransform'])

model.load_state_dict(torch.load(os.path.join(output_folder, model_name)))
model.set_actnorm_init()

model = model.to(device)
model = model.eval()



with torch.no_grad():
    images = model(y_onehot=None, temperature=1, reverse=True).cpu()  
    better_dup_images = model(y_onehot=None, temperature=1, z=model._last_z, reverse=True, use_last_split=True).cpu()   
    dup_images = model(y_onehot=None, temperature=1, z=model._last_z, reverse=True).cpu()   
    worse_dup_images = model(y_onehot=None, temperature=1, z=model._last_z, reverse=True).cpu()   

l2_err =  torch.pow((images - dup_images).view(images.shape[0], -1), 2).sum(-1).mean()
better_l2_err =  torch.pow((images - better_dup_images).view(images.shape[0], -1), 2).sum(-1).mean()
worse_l2_err =  torch.pow((images - worse_dup_images).view(images.shape[0], -1), 2).sum(-1).mean()
print(l2_err, better_l2_err, worse_l2_err)

f, axs = plt.subplots(1,4,figsize=(20,4))
grid = make_grid((postprocess(images)[:30]), nrow=6).permute(1,2,0)
axs[0].imshow(grid)
grid = make_grid((postprocess(dup_images)[:30]), nrow=6).permute(1,2,0)
axs[1].imshow(grid)
grid = make_grid((postprocess(better_dup_images)[:30]), nrow=6).permute(1,2,0)
axs[2].imshow(grid)
grid = make_grid((postprocess(worse_dup_images)[:30]), nrow=6).permute(1,2,0)
axs[3].imshow(grid)
for ax in axs:
	ax.axis('off')
plt.savefig(os.path.join(output_folder, 'recons.png'))