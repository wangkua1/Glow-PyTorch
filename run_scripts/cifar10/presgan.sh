DR=data
AMS=0
K=8
J=0
WG=1
SAVE=/scratch/ssd001/home/wangkuan/flow-gan
for LR in 5e-5; do
for DLR in 1e-4 5e-5; do
for exp in 0 1; do
for A in mine; do
for MLE in 0 1e-4; do
for E in 1e-4 1e-6; do
case "$exp" in
0)
    c=additive
    eps=0
;; 
1)
    c=gaffine
    eps=0
;;  
esac


# sbatch <<< \
# "#!/bin/bash


# #SBATCH --mem=10G
# #SBATCH -c 4
# #SBATCH --gres=gpu:1
# #SBATCH -p t4
# #SBATCH --time=24:00:00
# #SBATCH --output=/h/wangkuan/slurm/%j-out.txt
# #SBATCH --error=/h/wangkuan/slurm/%j-err.txt
# #SBATCH --qos=normal


# #necessary env
# source activate ebm


srun -p t4,p100 --gres=gpu:1 --mem=10G \
python train.py  \
    --fresh  \
    --gan  \
    --dataset cifar10  \
    --L 3  \
    --K ${K} \
    --hidden_channels 128  \
    --batch_size 64 \
    --n_init_batches 0 \
    --disc_lr ${DLR}  \
    --flow_permutation reverse   \
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
    --weight_entropy_reg $E \
    --output_dir ${SAVE}/cifar10/presgan/${MLE}-${A}-${exp}-${E}-${LR}-${DLR} \
    --eval_every 1000 \
    --optim_name adam \
    --svd_every 100000000000 \
    --no_warm_up 1 \
    --lr ${LR} \
    --actnorm_max_scale ${AMS} \
    --max_grad_clip  0  \
    --no_conv_actnorm 0 \
    --disc_arch $A \
    --no_learn_top &
# "
done
done
done
done
done
done
