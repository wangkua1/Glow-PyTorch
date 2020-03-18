K=8
LR=0
MLE=0
J=0
AMS=5
DR=/scratch/ssd001/home/wangkuan/data 

# python analyze_samples_interp.py  \
#     --fresh  \
#     --dataset mnist  \
#     --dataroot ${DR} \
#     --output_dir /scratch/ssd001/home/wangkuan/analysis-interp/additive-gan-only  \
#     --saved_model /scratch/ssd001/home/wangkuan/glow-icml20-2/gan-jac-reg/additive-0/ckpt_10000.pt \
#     --pgd_step_size 5 \
# 	--pgd_n_steps 500 \
# 	--pgd_f_project none \
# 	--pgd_l2_t 1e5 
    


python analyze_samples_interp.py  \
    --fresh  \
    --dataset mnist  \
    --dataroot ${DR} \
    --output_dir /scratch/ssd001/home/wangkuan/analysis-interp/no-split  \
    --saved_model /scratch/ssd001/home/wangkuan/glow/no-split/additive/ckpt_10000.pt \
    --pgd_step_size 5 \
	--pgd_n_steps 500 \
	--pgd_f_project none \
	--pgd_l2_t 1e5 