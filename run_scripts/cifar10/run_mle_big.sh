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
SAVE=/scratch/gobi1/wangkuan/flow-gan
for LR in 5e-4; do
for bs in 64; do
for LT in 0; do
for h in 512; do
for exp in 0 1 2 3 ; do
AMS=0
GP=0
db=0
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
    --db ${db} &


done
done
done
done
done
