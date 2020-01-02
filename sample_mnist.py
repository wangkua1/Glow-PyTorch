import json

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from datasets import get_CIFAR10, get_SVHN, get_MNIST, postprocess
from model import Glow
import ipdb
import os
device = torch.device("cuda")

output_folder = '/scratch/gobi2/wangkuan/glow/db-gan'
# output_folder = '/scratch/gobi2/wangkuan/glow/mnist-1x1-affine-512-1e-2'
model_name = 'glow_model_1.pth'

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

def sample(model):
    with torch.no_grad():
        assert not hparams['y_condition']
        y = None
        images = model(y_onehot=y, temperature=1, reverse=True, batch_size=32)
        # images = postprocess(model(y_onehot=y, temperature=1, reverse=True))

    return images.cpu()
# batch_size = 32
# images = []
# N = 10000
# iters = N//batch_size + 1
# for _ in range(iters):
# 	images.append(sample(model))

# # images =torch.stack(images)
# # images = postprocess(images[:N])
# # torch.save(images, '10k_samples.pt')
# # ipdb.set_trace()

images = sample(model)
# ipdb.set_trace()
grid = make_grid((postprocess(images)[:30]), nrow=6).permute(1,2,0)

plt.figure(figsize=(10,10))
plt.imshow(grid)
plt.axis('off')
plt.savefig(os.path.join(output_folder, 'sample.png'))