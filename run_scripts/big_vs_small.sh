

for name in big small; do
if [ $name == "big" ] 
then
    output_dir=/scratch/gobi1/wangkuan/glow/sota/$name
	DR=/scratch/gobi1/wangkuan/data
	h=512
	k=32
	p=invconv
	c=affine
	top=1
	lr=5e-4
else
    output_dir=/scratch/gobi1/wangkuan/glow/sota/$name
	DR=/scratch/gobi1/wangkuan/data
	h=128
	k=8
	p=reverse
	c=affine
	top=0
	lr=5e-4
fi

sbatch <<< \
"#!/bin/bash


#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=200:00:00
#SBATCH --output=/h/wangkuan/slurm/%j-out.txt
#SBATCH --error=/h/wangkuan/slurm/%j-err.txt
#SBATCH --qos=normal


#necessary env
source activate ebm


python train.py \
--dataroot $DR \
--dataset cifar10 \
--download \
--hidden_channels $h \
--K $k \
--flow_permutation $p  \
--flow_coupling $c \
--learn_top $top \
--output_dir $output_dir 
"
done

