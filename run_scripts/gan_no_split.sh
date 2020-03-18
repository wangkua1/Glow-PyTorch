DR=/scratch/ssd001/home/wangkuan/data
AMS=5
MLE=0
LR=5e-5
K=8
J=0

# python train.py  \
#     --fresh  \
#     --gan  \
#     --dataset mnist  \
#     --L 3  \
#     --K ${K} \
#     --hidden_channels 128  \
#     --batch_size 32 \
#     --n_init_batches 0 \
#     --disc_lr ${LR}  \
#     --flow_permutation reverse   \
#     --flow_coupling naffine  \
#     --affine_max_scale 5 \
#     --affine_scale_eps 2 \
#     --affine_eps .5 \
#     --weight_gan 1 \
#     --weight_prior ${MLE} \
#     --weight_logdet ${MLE} \
#     --jac_reg_lambda ${J} \
#     --flowgan 1 \
#     --dataroot ${DR} \
#     --output_dir /scratch/ssd001/home/wangkuan/glow/no-split/affine  \
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
#     --no_learn_top &


python train.py  \
    --fresh  \
    --gan  \
    --dataset mnist  \
    --L 3  \
    --K ${K}  \
    --hidden_channels 128  \
    --batch_size 32 \
    --n_init_batches 0 \
    --disc_lr ${LR}  \
    --flow_permutation reverse   \
    --flow_coupling additive  \
    --affine_max_scale 5 \
    --affine_scale_eps 2 \
    --affine_eps .5 \
    --weight_gan 1 \
    --weight_prior ${MLE} \
    --weight_logdet ${MLE} \
    --jac_reg_lambda ${J} \
    --flowgan 1 \
    --dataroot ${DR} \
    --output_dir /scratch/ssd001/home/wangkuan/glow/no-split/additive  \
    --eval_every 1000 \
    --optim_name adam \
    --svd_every 100000000000 \
    --no_warm_up 1 \
    --lr ${LR} \
    --clamp 0 \
    --no_actnorm 0 \
    --actnorm_max_scale ${AMS} \
    --init_sample 0 \
    --max_grad_clip  5  \
    --no_conv_actnorm 1 \
    --no_split 1 \
    --no_learn_top 