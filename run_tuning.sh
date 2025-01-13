#!/bin/bash

# Define hyperparameter values
learning_rates=(0.005 0.001 0.0005 0.0001)
batch_sizes=(2 4 8 16)
epochs=(500)

# Define static arguments
DATAROOT="/home/lweiss3/datasets/one_rectangle_building/"
CHECKPOINTS_DIR_BASE="/home/lweiss3/outputs/noise_modelling/tuning"
OUTPUT_DIR_BASE="/home/lweiss3/outputs/noise_modelling/tuning"
STATIC_ARGS="--dataroot $DATAROOT --name tuning --model pix2pix --input_nc 1 --output_nc 1 --dataset_mode noise --direction AtoB --verbose --resolution_512 --use_val_dataset --eval_epoch_freq 10 --mse_val_function"

# Iterate over hyperparameters
for lr in "${learning_rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for epoch in "${epochs[@]}"; do
            # Define dynamic checkpoint and output directories based on hyperparameters
            CHECKPOINTS_DIR="${CHECKPOINTS_DIR_BASE}/lr${lr}_bs${batch_size}_ep${epoch}"
            OUTPUT_DIR="${OUTPUT_DIR_BASE}/lr${lr}_bs${batch_size}_ep${epoch}"

            # Ensure directories exist
            mkdir -p "$CHECKPOINTS_DIR"
            mkdir -p "$OUTPUT_DIR"

            # Construct and run the command
            CMD="python /home/lweiss3/pytorch-CycleGAN-and-pix2pix/train.py $STATIC_ARGS --batch_size $batch_size --checkpoints_dir $CHECKPOINTS_DIR --out_val_results $OUTPUT_DIR --n_epochs $epoch --lr $lr"
            echo "Running: $CMD"
            eval "$CMD"
        done
    done
done