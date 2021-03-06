DR=data
AMS=5
# MLE=0
# LR=5e-5
K=16
SAVE=/scratch/gobi1/wangkuan/flow-gan
# for J in 10 1 0.1 0.01 0.001 0; do
for J in 0; do
for LR in 5e-5; do
for WG in 1; do
for LT in 0; do
if [ $WG == 0 ] 
then
    MLE=1
else
    MLE=0
fi
# srun -p gpu --gres=gpu:1 --mem=10G \
python train.py  \
    --fresh  \
    --gan  \
    --dataset cifar10  \
    --L 3  \
    --K ${K} \
    --hidden_channels 128  \
    --batch_size 64 \
    --n_init_batches 0 \
    --disc_lr ${LR}  \
    --flow_permutation reverse   \
    --flow_coupling naffine  \
    --affine_max_scale 5 \
    --affine_scale_eps 2 \
    --affine_eps .5 \
    --weight_gan ${WG} \
    --weight_prior ${MLE} \
    --weight_logdet ${MLE} \
    --jac_reg_lambda ${J} \
    --flowgan 1 \
    --dataroot ${DR} \
    --output_dir ${SAVE}/hyperp-tuning-db/affine-${WG}-${LR}-${LT}  \
    --eval_every 1000 \
    --optim_name adam \
    --svd_every 100000000000 \
    --no_warm_up 1 \
    --lr ${LR} \
    --clamp 0 \
    --no_actnorm 0 \
    --actnorm_max_scale ${AMS} \
    --init_sample 0 \
    --weight_entropy_reg 1 \
    --max_grad_clip  5  \
    --no_conv_actnorm 1 \
    --logittransform $LT \
    --disc_arch 'inv' \
    --no_learn_top 


# srun -p gpu --gres=gpu:1 --mem=10G \
# python train.py  \
#     --fresh  \
#     --gan  \
#     --dataset cifar10  \
#     --L 3  \
#     --K ${K}  \
#     --hidden_channels 128  \
#     --batch_size 32 \
#     --n_init_batches 0 \
#     --disc_lr ${LR}  \
#     --flow_permutation reverse   \
#     --flow_coupling additive  \
#     --affine_max_scale 5 \
#     --affine_scale_eps 2 \
#     --affine_eps .5 \
#     --weight_gan ${WG} \
#     --weight_prior ${MLE} \
#     --weight_logdet ${MLE} \
#     --jac_reg_lambda ${J} \
#     --flowgan 1 \
#     --dataroot ${DR} \
#     --output_dir ${SAVE}/hyperp-tuning/additive-${WG}-${LR}-${LT}  \
#     --eval_every 1000 \
#     --optim_name adam \
#     --svd_every 100000000000 \
#     --no_warm_up 1 \
#     --lr ${LR} \
#     --clamp 0 \
#     --no_actnorm 0 \
#     --actnorm_max_scale ${AMS} \
#     --init_sample 0 \
#     --max_grad_clip  5  \
#     --no_conv_actnorm 1 \
#     --logittransform $LT \
#     --no_learn_top &
done
done
done
done