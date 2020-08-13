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
AMS=0
GP=0
db=0
perm=reverse
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
    c=affine
    eps=0.01
    AMS=5
;;
4)
    c=affine
    eps=0
    AMS=0
    perm=invconv
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
;;
esac

cmd="train.py  \
    --fresh \
    --gan  \
    --dataset cifar10  \
    --L 3  \
    --K ${K} \
    --hidden_channels $h  \
    --batch_size $bs \
    --n_init_batches 10 \
    --disc_lr ${LR}  \
    --flow_permutation ${perm}   \
    --flow_coupling ${c}  \
    --affine_max_scale 5 \
    --affine_scale_eps 2 \
    --affine_eps ${eps} \
    --weight_gan ${WG} \
    --weight_prior ${MLE} \
    --weight_logdet ${MLE} \
    --jac_reg_lambda ${J} \
    --flowgan 1 \
    --dataroot ${DR} \
    --output_dir ${SAVE}/cifar10-mle-big/${c}-${exp} \
    --eval_every 1000 \
    --optim_name adamax \
    --svd_every 100000000000 \
    --no_warm_up 0 \
    --lr ${LR} \
    --max_grad_clip  ${GP}  \
    --no_conv_actnorm 0 \
    --actnorm_max_scale ${AMS} \
    --logittransform $LT \
    --epochs 500 \
    --db ${db}"


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
#SBATCH --qos=normal
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
