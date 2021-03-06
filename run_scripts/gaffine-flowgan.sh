
for LR in 1e-4 1e-5; do
for MLE in 0 0.1; do
srun --gres=gpu:1 -p gpu --mem=16G \
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
    --flow_coupling gaffine  \
    --affine_max_scale 5 \
    --affine_scale_eps 2 \
    --affine_eps .5 \
    --weight_gan 1 \
    --weight_prior ${MLE} \
    --weight_logdet 0${MLE} \
    --flowgan 1 \
    --dataroot /scratch/gobi1/wangkuan/data \
    --output_dir /scratch/gobi1/wangkuan/glow/gaffine-flowgan/${LR}-${MLE} \
    --eval_every 100 \
    --optim_name adam \
    --svd_every 100000000000 \
    --no_warm_up 1 \
    --lr ${LR} \
    --clamp 0 \
    --no_actnorm 0 \
    --actnorm_max_scale 5 \
    --max_grad_clip  5  \
    --no_conv_actnorm 1 \
    --no_learn_top &

srun --gres=gpu:1 -p gpu --mem=16G \
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
    --flow_coupling gaffine  \
    --affine_max_scale 5 \
    --affine_scale_eps 2 \
    --affine_eps .5 \
    --weight_gan 1 \
    --weight_prior ${MLE} \
    --weight_logdet 0${MLE} \
    --flowgan 1 \
    --dataroot /scratch/gobi1/wangkuan/data \
    --output_dir /scratch/gobi1/wangkuan/glow/gaffine-flowgan/logit-${LR}-${MLE} \
    --eval_every 100 \
    --optim_name adam \
    --svd_every 100000000000 \
    --no_warm_up 1 \
    --lr ${LR} \
    --clamp 0 \
    --no_actnorm 0 \
    --actnorm_max_scale 5 \
    --max_grad_clip 5 \
    --no_conv_actnorm 1 \
    --logittransform \
    --no_learn_top &
done
done

