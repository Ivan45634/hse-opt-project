#!/usr/bin/env bash
clear
for optimizer in adamw
do
    export CUDA_VISIBLE_DEVICES=3
    python ./src/run_experiment.py \
        --problem libsvm \
        --dataset mushrooms \
        --eval_runs 1 \
        --n_epoches_train 5 \
        --tune_runs 40 \
        --optimizer $optimizer \
        --hidden_dim 10 \
        --no_bias \
        --use_old_tune_params \
        --lmo spectral \
        --precondition_type norm \
        --weight_init uniform \
        --momentum 0.9 \
        --eps 1e-40 \
        --ns_steps 20 \
        --wandb \
        # --wandb # --use_old_tune_params \ --tune \
done
