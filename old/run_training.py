import os
import shutil
import tarfile
from diffusers import DiffusionPipeline
from cog import BaseModel, Input, Path

from predict import SDXL_MODEL_CACHE, download_weights
from preprocess import preprocess
from trainer_pti import main
import torch

"""
Wrapper around actual trainer.
"""


# TO VARY
input_images_list = ["input.zip", "input_albu.zip", "input_cropped.zip"]

# VARIATIONS 
RESOLUTIONS = [512, 768]
unet_learning_rates = [1e-6, 5e-5, 1e-5] 

MAX_TRAIN_STEPS = 4000   # checkpointing every 500 anyway

for RESOLUTION in RESOLUTIONS:
    for unet_learning_rate in unet_learning_rates:
        for input_images in input_images_list:

            OUTPUT_DIR = "training_out_" + str(RESOLUTION) + "_" + str(unet_learning_rate) + "_" + input_images.split(".")[0]
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

            token_string = "TOK"
            token_map = token_string + ":2"
            # input_images = "example_datasets/zeke.zip"
            # Process 'token_to_train' and 'input_data_tar_or_zip'
            inserting_list_tokens = token_map.split(",")

            token_dict = {}
            running_tok_cnt = 0
            all_token_lists = []
            for token in inserting_list_tokens:
                n_tok = int(token.split(":")[1])

                token_dict[token.split(":")[0]] = "".join(
                    [f"<s{i + running_tok_cnt}>" for i in range(n_tok)]
                )
                all_token_lists.extend([f"<s{i + running_tok_cnt}>" for i in range(n_tok)])

                running_tok_cnt += n_tok

            input_dir = preprocess(
                input_images_filetype='zip',
                input_zip_path=input_images,
                caption_text="",
                mask_target_prompts="dress",
                target_size=512,
                crop_based_on_salience=None,
                use_face_detection_instead=False,
                temp=1.0,
                substitution_tokens=list(token_dict.keys()),
            )
            print("input dir:", input_dir)


            SDXL_MODEL_CACHE = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
            main(
                pretrained_model_name_or_path=SDXL_MODEL_CACHE,
                instance_data_dir=os.path.join(input_dir, "captions.csv"),
                output_dir=OUTPUT_DIR,
                seed=1,
                resolution=RESOLUTION,
                train_batch_size=4,
                unet_learning_rate=unet_learning_rate,
                max_train_steps=MAX_TRAIN_STEPS,
                checkpointing_steps=500,
                scale_lr=False,
                max_grad_norm=1.0,
                allow_tf32=True,
                mixed_precision="bf16",
                device="cuda:0",
                is_lora=True,
                # num_train_epochs=num_train_epochs,
                # gradient_accumulation_steps=1,
                # ti_lr=ti_lr,
                # lora_lr=lora_lr,
                # lr_scheduler=lr_scheduler,
                # lr_warmup_steps=lr_warmup_steps,
                # token_dict=token_dict,
                # inserting_list_tokens=all_token_lists,
                # verbose=verbose,
                # lora_rank=,
            )