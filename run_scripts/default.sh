DR=/scratch/gobi1/wangkuan/data
h=512
k=32
p=invconv
c=affine
top=1
lr=5e-4
output_dir=/scratch/gobi1/wangkuan/glow/sota/default
python train.py \
--dataroot DR \
--dataset cifar10 \
--download \
--hidden_channels $h \
--K $k \
--flow_permutation $p  \
--flow_coupling $c \
--learn_top $top \
--output_dir $output_dir \
--fresh \
--db