#!/bin/bash

#SBATCH --error=logdir/log.preproc.%A_%a_.err
#SBATCH --output=logdir/log.preproc.%A_%a_.out
#SBATCH --job-name="preproc"
#SBATCH --gres=gpu:A40:1


cd /home/gshen/work_dir/unprobe
source .venv/bin/activate

uv run src/preprocessing.py --librispeech_split dev-clean
uv run src/preprocessing.py --librispeech_split train-clean-100