# useful commands


update W'u:

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 30-40 python utils/LoRA.py --prune_model jeffwan_llama_7b_hf_whitening_only_0.8.pt --data_path yahma/alpaca-cleaned --output_dir ./first_half --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64


CUDA_VISIBLE_DEVICES=0 taskset -c 30-40 python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora ./first_half /first_half --step 4
```


```bash
python utils/LoRA.py --prune_model ./first_half/merge.pt --data_path yahma/alpaca-cleaned --output_dir ./second_half --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora ./first_half /first_half --step 4

python SVDLLM.py --model_path ./first_half/merge.pt --lora ./second_half --step 4
```

