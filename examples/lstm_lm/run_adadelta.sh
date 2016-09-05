#!/usr/bin/env bash

#$ -cwd
#$ -S /bin/bash
#$ -M gkumar6@jhu.edu
#$ -m eas
#$ -l gpu=1
#$ -V
#$ -j y -o log/adadelta.log

. ../../setup.sh

THEANO_FLAGS=device=gpu python train.py \
    --save-to lm_adadelta.npz \
    --dataset ../../data/simple-examples/data
