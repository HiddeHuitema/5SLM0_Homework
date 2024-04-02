export WANDB_API_KEY=d7156e6b9496552b06075dfc9278a96a48a2a50e
export WANDB_DIR=wandb/$SLURM_JOBID
export WANDB_CONFIG_DIR=wandb/$SLURM_JOBID
export WANDB_CACHE_DIR=wandb/$SLURM_JOBID
export WANDB_START_METHOD="thread"
wandb login

torchrun --nnodes=1 --nproc_per_node=1 train.py \
         --data_path "/gpfs/work5/0/jhstue005/JHS_data/CityScapes" \
         --resizing_factor 8 \
         --n_workers 8
