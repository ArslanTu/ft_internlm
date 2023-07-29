# run

```bash
torchrun --nproc_per_node=8 train.py --data_name belle_open_source_500k --data_path ./data/Belle_open_source_0.5M.json --output_dir ./output --seed 42 --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 4 --logging_steps 10 --evaluation_strategy steps --save_strategy steps --eval_steps 100 --save_steps 500 --save_total_limit 20  --load_best_model_at_end true --optim adamw_torch --ddp_find_unused_parameters false
```