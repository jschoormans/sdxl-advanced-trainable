#!/bin/bash
instance_prompt="Woman wearing a TOK dress" # @param
validation_prompt="a black woman wearing a TOK dress, on the beach." # @param
rank=8 # @param

for use_dora in true false; do
  if [ "$use_dora" = true ]; then
    output_dir="black_dress_dora"
  else
    output_dir="black_dress_lora"
  fi

  accelerate launch train_dreambooth_lora_sdxl_advanced.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
    --dataset_name="./black_dress" \
    --instance_prompt="$instance_prompt" \
    --validation_prompt="$validation_prompt" \
    --output_dir="$output_dir" \
    --caption_column="prompt" \
    --mixed_precision="bf16" \
    --resolution=512 \
    --train_batch_size=3 \
    --repeats=1 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --learning_rate=1.0 \
    --text_encoder_lr=1.0 \
    --adam_beta2=0.99 \
    --optimizer="prodigy"\
    --train_text_encoder_ti\
    --train_text_encoder_ti_frac=0.5\
    --snr_gamma=5.0 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --rank="$rank" \
    --use_dora="$use_dora" \
    --max_train_steps=1000 \
    --checkpointing_steps=100 \
    --seed="0" \
    --report_to="wandb" \
    --num_validation_images=4
done
