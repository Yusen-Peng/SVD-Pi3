#!/bin/bash

# example of compressing LLaMA-7B with SVDLLM
FINE_TUNE_PATH="."
# run data whitening with 20% compression ratio
python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path .

python SVDLLM.py --step 4 --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt


# finetune the compressed model with lora
python utils/LoRA.py --prune_model  --data_path yahma/alpaca-cleaned --output_dir ./first_half --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora ./first_half /first_half --step 4
python utils/LoRA.py --prune_model ./first_half/merge.pt --data_path yahma/alpaca-cleaned --output_dir ./second_half --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora ./first_half /first_half --step 4
python SVDLLM.py --model_path ./first_half/merge.pt --lora ./second_half --step 4
