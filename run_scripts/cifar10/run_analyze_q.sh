
DR=data
# MLE=0
# LR=5e-5
K=32
J=0
WG=0
MLE=1
GP=0
AMS=0
db=0
SAVE=${ROOT1}/flow-gan
for LR in 5e-4; do
for bs in 64; do
for LT in 0; do
for h in 512; do
for exp in 4 6; do
for iter in 1000 5000 10000 20000 50000 100000; do
# for iter in 0 1000 2000 3000 4000 5000 10000 20000 50000 100000; do
AMS=0
GP=0
db=0
prefix=ckpt_
case "$exp" in
0)
    c=additive
    eps=0
;; 
1)
    c=affine
    eps=0.01
;;  
2)
    c=additive
    eps=0
    AMS=5
;;
3)
    c=naffine
    eps=0.1
;;
4)
    c=affine
    eps=0
    AMS=0
    perm=invconv
    prefix=ckpt_sd_
;;
5)
    c=naffine
    eps=0.5
;;
6)
    c=additive
    eps=0
    AMS=0
    perm=invconv
    prefix=ckpt_sd_
;;
esac

output_dir=${SAVE}/cifar10-mle-big/${c}-${exp}

cmd="analyze.py  \
    --dataset cifar10  \
    --dataroot ${DR} \
    --glow_path ${output_dir}/${prefix}${iter}.pt"


if [ $1 == 0 ] 
then
python $cmd
else
sbatch <<< \
"#!/bin/bash
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=200:00:00
#SBATCH --output=/h/wangkuan/slurm/%j-out.txt
#SBATCH --error=/h/wangkuan/slurm/%j-err.txt
#necessary env
source activate ebm
python $cmd
"
# srun -p gpu --mem=16G --gres=gpu:1 \
# python $cmd &
fi
done


done
done
done
done
done
done
