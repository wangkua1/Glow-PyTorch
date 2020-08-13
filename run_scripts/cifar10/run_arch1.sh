DR=data
AMS=5

SAVE=/scratch/gobi1/wangkuan/flow-gan
for J in 0; do
for DLR in 1e-4; do
for BS in 64; do
for K in 8; do
for A in mine; do
for LT in 0; do
for c in naffine; do
for exp in 0 1 2; do

case "$exp" in
0)
WG=1
MLE=0
J=0
;;
         
1)
WG=1
MLE=1e-3
J=0
;;
    
2)
WG=1
MLE=0
J=1e-3
;;

esac



# srun -p gpu --gres=gpu:1 --mem=10G \
python train.py  \
    --fresh  \
    --gan  \
    --dataset cifar10  \
    --L 3  \
    --K ${K} \
    --hidden_channels 128  \
    --batch_size 32 \
    --n_init_batches 0 \
    --disc_lr ${DLR}  \
    --flow_permutation reverse   \
    --flow_coupling ${c}  \
    --affine_max_scale 5 \
    --affine_scale_eps 2 \
    --affine_eps .5 \
    --weight_gan ${WG} \
    --weight_prior ${MLE} \
    --weight_logdet ${MLE} \
    --jac_reg_lambda ${J} \
    --flowgan 1 \
    --dataroot ${DR} \
    --output_dir ${SAVE}/quick/${c}-${A}-${exp}  \
    --eval_every 1000 \
    --optim_name adam \
    --svd_every 100000000000 \
    --no_warm_up 1 \
    --lr 5e-5 \
    --clamp 0 \
    --no_actnorm 0 \
    --actnorm_max_scale ${AMS} \
    --init_sample 0 \
    --max_grad_clip  5  \
    --no_conv_actnorm 1 \
    --logittransform $LT \
    --disc_arch $A \
    --no_learn_top \
    --epochs 100

done
done
done
done
done
done
done
done