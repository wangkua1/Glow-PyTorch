import json

import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import ipdb
from datasets import get_CIFAR10, get_SVHN
from model import Glow
import cv2
from datasets import postprocess
from anomaly import show_ood_detection_results_softmax
import numpy as np
device = torch.device("cuda")

output_folder = '/scratch/ssd001/home/wangkuan/glow/glow/'
model_name = 'glow_affine_coupling.pt'

with open(output_folder + 'hparams.json') as json_file:  
    hparams = json.load(json_file)
    
print(hparams)

image_shape, num_classes, _, test_cifar = get_CIFAR10(hparams['augment'], '/scratch/ssd001/home/wangkuan', True)
image_shape, num_classes, _, test_svhn = get_SVHN(hparams['augment'], '/scratch/ssd001/home/wangkuan', True)

model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])

model.load_state_dict(torch.load(output_folder + model_name))
model.set_actnorm_init()

model = model.to(device)

model = model.eval()



def batch_png_code_length(x, compression_level=3):
	assert x.max() <= .5 and x.min() >= -.5
	assert len(x.shape) == 4 and x.shape[1] in [1,3]
	x = postprocess(x) # convert into pixel domain
	code_lens = [len(cv2.imencode('.png', x[n].permute(1,2,0).cpu().numpy(), [cv2.IMWRITE_PNG_COMPRESSION, compression_level] )[1].tostring())/float(x[n].nelement()) for n in range(len(x))]
	return code_lens

cifar_iter = torch.utils.data.DataLoader(test_cifar, batch_size=512, num_workers=6).__iter__()
svhn_iter = torch.utils.data.DataLoader(test_svhn, batch_size=512, num_workers=6).__iter__()
cifar_nlls = []
cifar_code_lens = []
svhn_nlls = []
svhn_code_lens = []
for n in range(10):
	x = cifar_iter.__next__()[0].to(device)
	x = x.to(device)
	code_lens = batch_png_code_length(x)
	with torch.no_grad():
		nll = model(x)[1]
	cifar_nlls.append(nll)
	cifar_code_lens.append(code_lens)

	x = svhn_iter.__next__()[0].to(device)
	x = x.to(device)
	code_lens = batch_png_code_length(x)
	with torch.no_grad():
		nll = model(x)[1]
	svhn_nlls.append(nll)
	svhn_code_lens.append(code_lens)
nll_auroc = show_ood_detection_results_softmax(torch.cat(cifar_nlls).cpu().numpy(),
								   torch.cat(svhn_nlls).cpu().numpy())[1]
ll_auroc = show_ood_detection_results_softmax(-torch.cat(cifar_nlls).cpu().numpy(),
								   -torch.cat(svhn_nlls).cpu().numpy())[1]
s_auroc = show_ood_detection_results_softmax(torch.cat(cifar_nlls).cpu().numpy() - np.hstack(cifar_code_lens),
								   torch.cat(svhn_nlls).cpu().numpy() - np.hstack(svhn_code_lens)/3072)[1]



plt.scatter(code_lens, nll.cpu().numpy())
plt.savefig('code_lens_vs_nll.jpeg')