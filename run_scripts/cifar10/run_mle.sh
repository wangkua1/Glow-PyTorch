DR=data
# MLE=0
# LR=5e-5
K=8
J=0
WG=0
MLE=1
GP=0
AMS=0
db=0
SAVE=/scratch/gobi1/wangkuan/flow-gan
for LR in 5e-4; do
for bs in 64; do
for LT in 0; do
for h in 128; do
for exp in 4 5 6; do

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
    c=gaffine
    eps=0
;;
3)
    c=naffine
    eps=0.1
;;
4)
    c=affine
    eps=0.01
    GP=5
    db=1
;;
5)
    c=affine
    eps=0.01
    AMS=5
    db=1
;;
6)
    c=affine
    eps=0.01
    AMS=5
    GP=5
    db=1
;;
esac

srun -p gpu --gres=gpu:1 --mem=10G \
python train.py  \
    --fresh  \
    --gan  \
    --dataset cifar10  \
    --L 3  \
    --K ${K} \
    --hidden_channels $h  \
    --batch_size $bs \
    --n_init_batches 10 \
    --disc_lr ${LR}  \
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
    --output_dir ${SAVE}/cifar10-mle-db1/${c}-${exp} \
    --eval_every 1000 \
    --optim_name adamax \
    --svd_every 100000000000 \
    --no_warm_up 0 \
    --lr ${LR} \
    --max_grad_clip  ${GP}  \
    --no_conv_actnorm 0 \
    --actnorm_max_scale ${AMS} \
    --logittransform $LT \
    --no_learn_top \
    --db ${db} &


done
done
done
done
done
