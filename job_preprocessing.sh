#!/bin/bash

#SBATCH --error=logdir/log.preproc.%A.err
#SBATCH --output=logdir/log.preproc.%A.out
#SBATCH --job-name="preproc"
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00


cd /home/gshen/work_dir/unprobe
source .venv/bin/activate

# uv run src/preprocessing.py --librispeech_split dev-clean --overwrite
srun python src/preprocessing.py --do_base_only --overwrite_base --overwrite_textgrid --librispeech_split train-clean-100
srun python src/preprocessing.py --librispeech_split train-clean-100 --modelname "facebook/wav2vec2-base" --overwrite
