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
OUTPUT_DIR = "training_out"


class TrainingOutput(BaseModel):
    weights: Path


from typing import Tuple
import subprocess


def train(
    input_images: Path = Input(
        description="A .zip or .tar file containing the image files that will be used for fine-tuning"
    ),
    # seed: int = Input(
    #     description="Random seed for reproducible training. Leave empty to use a random seed",
    #     default=None,
    # ),
    # # resolution: int = Input(
    # #     description="Square pixel resolution which your images will be resized to for training",
    # #     default=768,
    # # # ),
    # # # train_batch_size: int = Input(
    # # #     description="Batch size (per device) for training",
    # # #     default=4,
    # # # ),
    # # # num_train_epochs: int = Input(
    # # #     description="Number of epochs to loop through your training dataset",
    # # #     default=4000,
    # # # ),
    # # # max_train_steps: int = Input(
    # # #     description="Number of individual training steps. Takes precedence over num_train_epochs",
    # # #     default=1000,
    # # # ),
    # gradient_accumulation_steps: int = Input(
    #     description="Number of training steps to accumulate before a backward pass. Effective batch size = gradient_accumulation_steps * batch_size",
    #     default=1,
    # ), # todo.
    # is_lora: bool = Input(
    #     description="Whether to use LoRA training. If set to False, will use Full fine tuning",
    #     default=True,
    # ),
    # unet_learning_rate: float = Input(
    #     description="Learning rate for the U-Net. We recommend this value to be somewhere between `1e-6` to `1e-5`.",
    #     default=1e-6,
    # ),
    # ti_lr: float = Input(
    #     description="Scaling of learning rate for training textual inversion embeddings. Don't alter unless you know what you're doing.",
    #     default=3e-4,
    # ),
    # lora_lr: float = Input(
    #     description="Scaling of learning rate for training LoRA embeddings. Don't alter unless you know what you're doing.",
    #     default=1e-4,
    # ),
    # lora_rank: int = Input(
    #     description="Rank of LoRA embeddings. Don't alter unless you know what you're doing.",
    #     default=32,
    # ),
    # lr_scheduler: str = Input(
    #     description="Learning rate scheduler to use for training",
    #     default="constant",
    #     choices=[
    #         "constant",
    #         "linear",
    #     ],
    # ),
    # lr_warmup_steps: int = Input(
    #     description="Number of warmup steps for lr schedulers with warmups.",
    #     default=100,
    # ),
    token_string: str = Input(
        description="A unique string that will be trained to refer to the concept in the input images. Can be anything, but TOK works well",
        default="TOK",
    ),
    # token_map: str = Input(
    #     description="String of token and their impact size specificing tokens used in the dataset. This will be in format of `token1:size1,token2:size2,...`.",
    #     default="TOK:2",
    # ),
    # caption_prefix: str = Input(
    #     description="Text which will be used as prefix during automatic captioning. Must contain the `token_string`. For example, if caption text is 'a photo of TOK', automatic captioning will expand to 'a photo of TOK under a bridge', 'a photo of TOK holding a cup', etc.",
    #     default="a photo of TOK, ",
    # ),
    mask_target_prompts: str = Input(
        description="Prompt that describes part of the image that you will find important. For example, if you are fine-tuning your pet, `photo of a dog` will be a good prompt. Prompt-based masking is used to focus the fine-tuning process on the important/salient parts of the image",
        default=None,
    ),
    # crop_based_on_salience: bool = Input(
    #     description="If you want to crop the image to `target_size` based on the important parts of the image, set this to True. If you want to crop the image based on face detection, set this to False",
    #     default=True,
    # ),
    # use_face_detection_instead: bool = Input(
    #     description="If you want to use face detection instead of CLIPSeg for masking. For face applications, we recommend using this option.",
    #     default=False,
    # ),
    # clipseg_temperature: float = Input(
    #     description="How blurry you want the CLIPSeg mask to be. We recommend this value be something between `0.5` to `1.0`. If you want to have more sharp mask (but thus more errorful), you can decrease this value.",
    #     default=1.0,
    # ),
    # verbose: bool = Input(description="verbose output", default=True),
    # checkpointing_steps: int = Input(
    #     description="Number of steps between saving checkpoints. Set to very very high number to disable checkpointing, because you don't need one.",
    #     default=999999,
    # ),
    # input_images_filetype: str = Input(
    #     description="Filetype of the input images. Can be either `zip` or `tar`. By default its `infer`, and it will be inferred from the ext of input file.",
    #     default="infer",
    #     choices=["zip", "tar", "infer"],
    # ),
) -> TrainingOutput:
    # Hard-code token_map for now. Make it configurable once we support multiple concepts or user-uploaded caption csv.
    token_map = token_string + ":2"

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
        input_images_filetype="infer",
        input_zip_path=input_images,
        caption_text="a photo of TOK,",
        mask_target_prompts=mask_target_prompts,
        target_size=512,
        crop_based_on_salience=True,
        use_face_detection_instead=False,
        temp=1.0,
        substitution_tokens=list(token_dict.keys()),
    )
    
    import pandas as pd
    df = pd.read_csv(os.path.join(input_dir, "captions.csv"))
    df = df.rename(columns={"file_name": "file_name", "caption": "prompt"})
    df.to_json(os.path.join(input_dir, "metadata.jsonl"), orient="records", lines=True)
    
    # remove captions.csv
    os.remove(os.path.join(input_dir, "captions.csv"))
    
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    
    if not os.path.exists(SDXL_MODEL_CACHE):
        DiffusionPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0",
            torch_dtype=torch.float16,
        ).save_pretrained(SDXL_MODEL_CACHE, safe_serialization=True)

    # TO DO : ADD METADATA
    instance_prompt = "a photo of a TOK dress"
    validation_prompt = "a person wearing a TOK dress"

    command = [
        "accelerate",
        "launch",
        "train_dreambooth_lora_sdxl_advanced.py",
        f"--pretrained_model_name_or_path={SDXL_MODEL_CACHE}",
        f"--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
        f"--dataset_name={input_dir}",
        f"--instance_prompt={instance_prompt}",
        f"--validation_prompt={validation_prompt}",
        f"--output_dir={OUTPUT_DIR}",
        "--caption_column=prompt",
        "--mixed_precision=bf16",
        "--resolution=512",
        "--train_batch_size=3",
        "--repeats=1",
        "--gradient_accumulation_steps=1",
        "--gradient_checkpointing",
        "--learning_rate=1.0",
        "--text_encoder_lr=1.0",
        "--adam_beta2=0.99",
        "--optimizer=prodigy",
        "--train_text_encoder_ti",
        "--train_text_encoder_ti_frac=0.5",
        "--snr_gamma=5.0",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--rank=$rank",
        "--use_dora=$use_dora",
        "--max_train_steps=1000",
        "--checkpointing_steps=100",
        "--seed=0",
        "--report_to=wandb",
        "--num_validation_images=4"
    ]

    subprocess.run(command, check=True)

    

    # main(
    #     pretrained_model_name_or_path=SDXL_MODEL_CACHE,
    #     instance_data_dir=os.path.join(input_dir, "captions.csv"),
    #     output_dir=OUTPUT_DIR,
    #     seed=seed,
    #     resolution=resolution,
    #     train_batch_size=train_batch_size,
    #     num_train_epochs=num_train_epochs,
    #     max_train_steps=max_train_steps,
    #     gradient_accumulation_steps=1,
    #     unet_learning_rate=unet_learning_rate,
    #     ti_lr=ti_lr,
    #     lora_lr=lora_lr,
    #     lr_scheduler=lr_scheduler,
    #     lr_warmup_steps=lr_warmup_steps,
    #     token_dict=token_dict,
    #     inserting_list_tokens=all_token_lists,
    #     verbose=verbose,
    #     checkpointing_steps=checkpointing_steps,
    #     scale_lr=False,
    #     max_grad_norm=1.0,
    #     allow_tf32=True,
    #     mixed_precision="bf16",
    #     device="cuda:0",
    #     lora_rank=lora_rank,
    #     is_lora=is_lora,
    # )

    directory = Path(OUTPUT_DIR)
    out_path = "trained_model.tar"

    with tarfile.open(out_path, "w") as tar:
        for file_path in directory.rglob("*"):
            print(file_path)
            arcname = file_path.relative_to(directory)
            tar.add(file_path, arcname=arcname)

    return TrainingOutput(weights=Path(out_path))
