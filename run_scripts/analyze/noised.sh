K=8
LR=0
MLE=0
J=0
AMS=5
DR=/scratch/ssd001/home/wangkuan/data 


# python analyze_samples_noised.py  \
#     --fresh  \
#     --dataset mnist  \
#     --dataroot ${DR} \
#     --output_dir /scratch/ssd001/home/wangkuan/analysis-noised/additive-gan-only  \
#     --saved_model /scratch/ssd001/home/wangkuan/glow-icml20-2/gan-jac-reg/additive-0/ckpt_10000.pt 
    

python analyze_samples_noised.py  \
    --fresh  \
    --dataset mnist  \
    --dataroot ${DR} \
    --output_dir /scratch/ssd001/home/wangkuan/analysis-noised/db \
    --saved_model /scratch/ssd001/home/wangkuan/glow-icml20-2/gan-jac-reg/additive-0/ckpt_10000.pt 
    