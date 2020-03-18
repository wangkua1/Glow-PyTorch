K=8
LR=0
MLE=0
J=0
AMS=5
DR=/scratch/ssd001/home/wangkuan/data 

# python analyze_samples.py  \
#     --fresh  \
#     --dataset mnist  \
#     --dataroot ${DR} \
#     --output_dir /scratch/ssd001/home/wangkuan/analysis/additive  \
#     --saved_model /scratch/ssd001/home/wangkuan/glow-icml20-2/gan-mle/additive-1/ckpt_10000.pt 

# python analyze_samples.py  \
#     --fresh  \
#     --dataset mnist  \
#     --dataroot ${DR} \
#     --output_dir /scratch/ssd001/home/wangkuan/analysis/additive-gan-only  \
#     --saved_model /scratch/ssd001/home/wangkuan/glow-icml20-2/gan-jac-reg/additive-0/ckpt_10000.pt 

python analyze_samples.py  \
    --fresh  \
    --dataset mnist  \
    --dataroot ${DR} \
    --output_dir /scratch/ssd001/home/wangkuan/analysis/no-split/noised  \
    --saved_model /scratch/ssd001/home/wangkuan/glow/no-split/additive/ckpt_10000.pt 
        