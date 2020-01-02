
for SN in 0 1; do
    for C in additive affine; do 
        for H in 128; do
            for K in 32; do
# GAN
srun --gres=gpu:1 -p gpu --mem=16G \
python train.py  \
    --fresh  \
    --gan  \
    --dataset mnist  \
    --L 3  \
    --K ${K} \
    --hidden_channels ${H}  \
    --batch_size 32 \
    --lr 1e-5  \
    --disc_lr 1e-4 \
    --flow_permutation reverse   \
    --flow_coupling ${C}  \
    --no_learn_top \
    --sn ${SN} \
    --output_dir /scratch/gobi2/wangkuan/glow/rebuttal-guess/gan-${H}-${K}-${C}-${SN} &

# MLE 
srun --gres=gpu:1 -p gpu --mem=16G \
python train.py  \
    --fresh  \
    --dataset mnist  \
    --L 3  \
    --K ${K} \
    --hidden_channels ${H}  \
    --batch_size 32 \
    --lr 1e-3  \
    --flow_permutation reverse   \
    --flow_coupling ${C}  \
    --sn ${SN} \
    --output_dir /scratch/gobi2/wangkuan/glow/rebuttal-guess/mle-${H}-${K}-${C}-${SN} &
            done
        done
    done
done
