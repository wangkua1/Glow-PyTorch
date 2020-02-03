
for LR in 5e-5; do
for MLE in 0 1; do
for AE in 0 1e-3 1e-1; do
srun --gres=gpu:1 --mem=12G \
-p gpu --account=deadline --qos=deadline \
python train.py  \
    --fresh  \
    --gan  \
    --dataset mnist  \
    --L 3  \
    --K 16 \
    --hidden_channels 128  \
    --batch_size 32 \
    --n_init_batches 20 \
    --disc_lr ${LR}  \
    --flow_permutation reverse   \
    --flow_coupling affine  \
    --affine_max_scale 5 \
    --affine_scale_eps 2 \
    --affine_eps .5 \
    --weight_gan 1 \
    --weight_prior ${MLE} \
    --weight_logdet ${MLE} \
    --flowgan 1 \
    --dataroot /scratch/ssd001/home/wangkuan/data \
    --output_dir /scratch/ssd001/home/wangkuan/glow/affine-flowgan-eps/${LR}-${MLE}-${AE} \
    --eval_every 2000 \
    --optim_name adam \
    --svd_every 100000000000 \
    --no_warm_up 1 \
    --lr ${LR} \
    --clamp 0 \
    --no_actnorm 0 \
    --actnorm_max_scale 0 \
    --max_grad_clip  5  \
    --actnorm_eps ${AE} \
    --no_conv_actnorm 1 \
    --no_learn_top &

srun --gres=gpu:1 --mem=12G \
-p gpu --account=deadline --qos=deadline \
python train.py  \
    --fresh  \
    --gan  \
    --dataset mnist  \
    --L 3  \
    --K 16 \
    --hidden_channels 128  \
    --batch_size 32 \
    --n_init_batches 20 \
    --disc_lr ${LR}  \
    --flow_permutation reverse   \
    --flow_coupling affine  \
    --affine_max_scale 5 \
    --affine_scale_eps 2 \
    --affine_eps .5 \
    --weight_gan 1 \
    --weight_prior ${MLE} \
    --weight_logdet ${MLE} \
    --flowgan 1 \
    --dataroot /scratch/ssd001/home/wangkuan/data \
    --output_dir /scratch/ssd001/home/wangkuan/glow/affine-flowgan-eps/logit-${LR}-${MLE}-${AE} \
    --eval_every 2000 \
    --optim_name adam \
    --svd_every 100000000000 \
    --no_warm_up 1 \
    --lr ${LR} \
    --clamp 0 \
    --no_actnorm 0 \
    --actnorm_max_scale 0 \
    --max_grad_clip  5  \
    --actnorm_eps ${AE} \
    --no_conv_actnorm 1 \
    --logittransform \
    --no_learn_top &
done
done
done
