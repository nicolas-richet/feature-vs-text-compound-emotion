#!/bin/bash


# Run the Python script with arguments
python3 text_emotion_classification.py  --gpu_id 2 --dataset 'C-EXPR-DB' --used_folds 0 1 2 3 4 --used_modalities 'V' --batch_size 4 --lr 3e-3 --max_grad_norm 0.3 --weight_decay 0.01 --lora_r 8 --lora_alpha 8 --lora_dropout 0.05 --training_method 'windows' --window_size 20 --window_hop 10 --epochs 100

