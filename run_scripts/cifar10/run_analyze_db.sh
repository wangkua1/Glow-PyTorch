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
SAVE=${ROOT1}/flow-gan
for LR in 5e-4; do
for bs in 64; do
for LT in 0; do
for h in 512; do
for exp in 0 ; do
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

# output_dir=${SAVE}/cifar10-mle-big/${c}-${exp}
output_dir=${SAVE}/cifar10-mle-small-db/${c}-${exp}

cmd="analyze.py  \
    --dataset cifar10  \
    --dataroot ${DR} \
    --glow_path ${output_dir}/ckpt_0.pt"


if [ $1 == 0 ] 
then
python $cmd
else
sbatch <<< \
"#!/bin/bash
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -p p100,t4
#SBATCH --time=200:00:00
#SBATCH --output=/h/wangkuan/slurm/%j-out.txt
#SBATCH --error=/h/wangkuan/slurm/%j-err.txt
#SBATCH --qos=normal
#necessary env
source activate ebm
python $cmd
"
# srun -p gpu --mem=16G --gres=gpu:1 \
# python $cmd &
fi
done



done
done
done
done
done
