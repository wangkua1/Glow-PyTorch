import json
import ipdb
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from datasets import get_CIFAR10, get_SVHN,get_CIFAR100
from model import Glow
import sklearn.metrics
import numpy as np

device = torch.device("cuda")

output_folder = 'glow/'
model_name = 'glow_affine_coupling.pt'

with open(output_folder + 'hparams.json') as json_file:  
    hparams = json.load(json_file)
    
print(hparams)
hparams['dataroot'] = '../mutual-information'

image_shape, num_classes, _, test_cifar = get_SVHN(hparams['augment'], hparams['dataroot'], hparams['download'])
# image_shape, num_classes, _, test_cifar = get_CIFAR10(hparams['augment'], hparams['dataroot'], hparams['download'])
image_shape, num_classes, _, test_svhn = get_CIFAR100(hparams['augment'], hparams['dataroot'], hparams['download'])

model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])

model.load_state_dict(torch.load(output_folder + model_name))
model.set_actnorm_init()

model = model.to(device)

model = model.eval()

# dataset = test_cifar
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=6)
# for x,y in dataloader:
# 	break

# x = x.to(device)
# y = y.to(device)

# with torch.no_grad():
# 	zs, nll, logits = model(x, y_onehot=y)

# with torch.no_grad():
# 	recon = model(z=zs, reverse=True)
# ipdb.set_trace()


def compute_nll(dataset, model):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=6)
    
    nlls = []
    for x,y in dataloader:
        x = x.to(device)
        
        if hparams['y_condition']:
            y = y.to(device)
        else:
            y = None
        
        with torch.no_grad():
            # _, nll, _ = model(x)
            _, nll, _ = model(x, y_onehot=y)
            nlls.append(nll)
        
    return torch.cat(nlls).cpu()


cifar_nll = compute_nll(test_cifar, model)
svhn_nll = compute_nll(test_svhn, model)

print("CIFAR NLL", torch.mean(cifar_nll))
print("SVHN NLL", torch.mean(svhn_nll))
ipdb.set_trace()
scores = np.concatenate([cifar_nll[:10000], svhn_nll][:10000])
labels = np.concatenate([np.ones_like(cifar_nll)[:10000], np.zeros_like(svhn_nll)[:10000]])
score = sklearn.metrics.roc_auc_score(labels, scores)

plt.figure()
plt.title("Histogram Glow - trained on CIFAR10")
plt.xlabel("Negative bits per dimension")
plt.hist(-svhn_nll.numpy(), label="CIFAR100", density=True, bins=30, alpha=.5)
plt.hist(-cifar_nll.numpy(), label="CIFAR10", density=True, bins=50, alpha=.5)
plt.legend()
plt.savefig('svhn.png')

