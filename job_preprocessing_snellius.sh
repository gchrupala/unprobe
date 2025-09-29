#!/bin/bash

#SBATCH --error=logdir/log.preproc.%A_%a.err
#SBATCH --output=logdir/log.preproc.%A_%a.out
#SBATCH --job-name="preproc"
#SBATCH --partition=gpu_mig
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=g.shen@tilburguniversity.edu
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00

cd /home/gshen/work_dir/unprobe
source snellius_modules
source .venv/bin/activate

uv run --extra cu126 src/preprocessing.py --librispeech_split train-clean-100 --overwrite