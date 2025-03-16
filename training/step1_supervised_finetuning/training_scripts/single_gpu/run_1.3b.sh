#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 1 training/step1_supervised_finetuning/main.py \
   --model_name_or_path ~/Work_space/models/opt-1.3b \
   --train_data_path ./processed_data/train.jsonl \
   --valid_data_path ./processed_data/valid.jsonl \
   --num_train_samples 90913 \
   --gradient_accumulation_steps 2 \
   --lora_dim 128 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT &> $OUTPUT/training.log \
