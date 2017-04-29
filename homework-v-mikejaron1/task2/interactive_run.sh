#!/bin/sh
srun --pty -t 0-02:00:00 --gres=gpu:1 -A edu /bin/bash
module load cuda80/toolkit cuda80/blas cudnn/5.1
module load anaconda/2-4.2.0

python task2.py
