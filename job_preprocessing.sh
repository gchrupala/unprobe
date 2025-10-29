#!/bin/bash

#SBATCH --error=logdir/log.preproc.%A.err
#SBATCH --output=logdir/log.preproc.%A.out
#SBATCH --error=logdir/log.preproc.%A.err
#SBATCH --output=logdir/log.preproc.%A.out
#SBATCH --job-name="preproc"
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00


cd /home/gshen/work_dir/unprobe
source .venv/bin/activate

export PATH="/home/gshen/.pixi/bin:/home/gshen/.local/bin:$PATH"
source ~/.pixi/completions/bash/*


# srun uv run --extra cu118 src/preprocessing.py --librispeech_split dev-clean --modelname "facebook/wav2vec2-base"  --overwrite
srun python src/preprocessing.py --librispeech_split "train-clean-100" --do_base
srun python src/preprocessing.py --librispeech_split train-clean-100 --do_transformer
