

srun --gres=gpu:1 -p gpu --mem=16G \
python train.py  \
    --fresh  \
    --gan  \
    --dataset mnist  \
    --L 3  \
    --K 16 \
    --hidden_channels 128  \
    --batch_size 32 \
    --lr 1e-5 \
    --disc_lr 1e-4 \
    --flow_permutation reverse   \
    --flow_coupling affine  \
    --no_learn_top \
    --weight_gan 0 \
    --weight_prior 1 \
    --weight_logdet 1  \
    --flowgan 1 \
    --eval_every 100 \
    --output_dir /scratch/gobi2/wangkuan/glow/affine-gan-mle/mle &

srun --gres=gpu:1 -p gpu --mem=16G \
python train.py  \
    --fresh  \
    --gan  \
    --dataset mnist  \
    --L 3  \
    --K 16 \
    --hidden_channels 128  \
    --batch_size 32 \
    --lr 1e-5 \
    --disc_lr 1e-4 \
    --flow_permutation reverse   \
    --flow_coupling affine  \
    --no_learn_top \
    --weight_gan 1 \
    --weight_prior 0 \
    --weight_logdet 0  \
    --flowgan 1 \
    --eval_every 100 \
    --output_dir /scratch/gobi2/wangkuan/glow/affine-gan-mle/gan &

for R in 1e-1 1e-2 1e-3 1e-4 1e-5; do
srun --gres=gpu:1 -p gpu --mem=16G \
python train.py  \
    --fresh  \
    --gan  \
    --dataset mnist  \
    --L 3  \
    --K 16 \
    --hidden_channels 128  \
    --batch_size 32 \
    --lr 1e-5 \
    --disc_lr 1e-4 \
    --flow_permutation reverse   \
    --flow_coupling affine  \
    --no_learn_top \
    --weight_gan 1 \
    --weight_prior ${R} \
    --weight_logdet ${R} \
    --flowgan 1 \
    --eval_every 100 \
    --output_dir /scratch/gobi2/wangkuan/glow/affine-gan-mle/gan+mle${R} &
done
