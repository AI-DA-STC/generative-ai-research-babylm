#!/bin/bash

MODELPATH=$1

for task in {boolq,cola,mnli,mrpc,multirc,qnli,qqp,rte,sst2,wsc}; do
    python train_lora.py \
        $MODELPATH \
        $task \
        --batch_size 64 \
        --num_epochs 32 \
        --learning_rate 3e-4 \
        --max_length 128
done
