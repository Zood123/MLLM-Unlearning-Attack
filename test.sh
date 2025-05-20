export HF_HOME= 
export HF_TOKEN= 

CUDA_VISIBLE_DEVICES=0,1 python llava_vlm_attack.py \
    --model_id vanilla_model_id \
    --model_path unlearned_model \
    --gpu_id 0 \
    --n_iters 1000 \
    --alpha 1 \
    --batch_size 6 \
    --save_dir adv_attack\