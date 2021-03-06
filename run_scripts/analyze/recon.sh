K=8
LR=0
MLE=0
J=0
AMS=5
DR=/scratch/ssd001/home/wangkuan/data 

python analyze_samples_recon.py  \
    --fresh  \
    --dataset mnist  \
    --dataroot ${DR} \
    --output_dir /scratch/ssd001/home/wangkuan/analysis-recon/no-split  \
    --saved_model /scratch/ssd001/home/wangkuan/glow/no-split/additive/ckpt_10000.pt  \
    --pgd_step_size 10 \
	--pgd_n_steps 1000 \
	--pgd_f_project none \
	--pgd_l2_t 1e5 
    