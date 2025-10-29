#!/bin/bash

#SBATCH --error=logdir/log.preproc.%A.err
#SBATCH --output=logdir/log.preproc.%A.out
#SBATCH --job-name="preproc"
#SBATCH --partition=gpu_a100
##SBATCH --partition=rome
#SBATCH --cpus-per-task=18
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=g.shen@tilburguniversity.edu
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00

cd /home/gshen/work_dir/unprobe
source snellius_modules
source .venv/bin/activate

#LIBRISPEECH_SPLIT="dev-clean"
LIBRISPEECH_SPLIT="train-clean-100"

srun python src/preprocessing.py --librispeech_split "$LIBRISPEECH_SPLIT" --do_base # --overwrite_textgrid
srun python src/preprocessing.py --librispeech_split "$LIBRISPEECH_SPLIT" --modelname "facebook/wav2vec2-base" --do_transformer
