import json

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from datasets import get_CIFAR10, get_SVHN, postprocess
from model import Glow
import ipdb
device = torch.device("cuda")

output_folder = 'glow/'
model_name = 'glow_affine_coupling.pt'

with open(output_folder + 'hparams.json') as json_file:  
    hparams = json.load(json_file)
hparams['dataroot'] = '../mutual-information'
  
image_shape, num_classes, _, test_cifar = get_CIFAR10(hparams['augment'], hparams['dataroot'], hparams['download'])
image_shape, num_classes, _, test_svhn = get_SVHN(hparams['augment'], hparams['dataroot'], hparams['download'])


model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])

model.load_state_dict(torch.load(output_folder + model_name))
model.set_actnorm_init()

model = model.to(device)

model = model.eval()

def sample(model):
    with torch.no_grad():
        if hparams['y_condition']:
            y = torch.eye(num_classes)
            y = y.repeat(batch_size // num_classes + 1)
            y = y[:32, :].to(device) # number hardcoded in model for now
        else:
            y = None

        images = model(y_onehot=y, temperature=1, reverse=True)
        # images = postprocess(model(y_onehot=y, temperature=1, reverse=True))

    return images.cpu()
batch_size = 32
images = []
N = 10000
iters = N//batch_size + 1
for _ in range(iters):
	images.append(sample(model))

images =torch.stack(images)
images = postprocess(images[:N])
torch.save(images, '10k_samples.pt')
ipdb.set_trace()

# images = sample(model)
# grid = make_grid(postprocess(images[:30]), nrow=6).permute(1,2,0)

# plt.figure(figsize=(10,10))
# plt.imshow(grid)
# plt.savefig('sample.png')
# plt.axis('off')