clear
#for dataset in cola mnli mrpc qnli qqp rte sst2 stsb
for r in 4 8
do
    # for lr in 1e-6 3e-5 5e-5 8e-5 1e-4 3e-4 5e-4 8e-4 1e-3
    for lr in tuned
    do
        export CUDA_VISIBLE_DEVICES=3
        python ./glue_experiment/run_glue.py \
            --dataset cola \
            --model meta-llama/Meta-Llama-3.1-8B \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate $lr \
            --lr_scheduler_type cosine \
            --max_seq_length 1024 \
            --warmup_steps 50 \
            --max_steps 1024 \
            --eval_strategy steps \
            --eval_step 64 \
            --max_val_samples 1001 \
            --save_strategy no \
            --ft_strategy LoRA \
            --lora_r $r \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --seed 18 \
            --do_eval true \
            --do_predict false \
            --bf16 true \
            --report_to wandb # none or wandb
    done
done
