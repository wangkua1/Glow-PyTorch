
for lr in 5e-5 1e-5; do 
    for disc_lr in 1e-4; do
        for H in 128 512; do
            for K in 16 32; do
srun --gres=gpu:1 -p gpu --mem=16G \
python train.py  \
    --fresh  \
    --gan  \
    --dataset mnist  \
    --L 3  \
    --K ${K} \
    --hidden_channels ${H}  \
    --batch_size 32 \
    --lr ${lr}  \
    --disc_lr ${disc_lr} \
    --flow_permutation reverse   \
    --flow_coupling additive  \
    --no_learn_top \
    --output_dir /scratch/gobi2/wangkuan/glow/gans1-1/simple-${H}-${K}-${lr}-${disc_lr} &

# srun --gres=gpu:1 -p gpu --mem=16G \
# python train.py  \
#     --fresh  \
#     --gan  \
#     --dataset mnist  \
#     --L 3  \
#     --K ${K} \
#     --hidden_channels ${H}  \
#     --batch_size 32 \
#     --lr ${lr}  \
#     --disc_lr ${disc_lr} \
#     --flow_permutation reverse   \
#     --flow_coupling additive  \
#     --no_learn_top \
#     --logittransform \
#     --output_dir /scratch/gobi2/wangkuan/glow/gans1-1/logit-${H}-${K}-${lr}-${disc_lr} &
            done
        done
    done
done
