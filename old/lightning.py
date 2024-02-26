import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, StableDiffusionXLControlNetPipeline, ControlNetModel
from huggingface_hub import hf_hub_download
from dataset_and_utils import TokenEmbeddingsHandler
from safetensors.torch import load_file
import json
import os
from diffusers.utils import load_image
from diffusers import AutoencoderKL



#   --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
#   --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \

base = "stabilityai/stable-diffusion-xl-base-1.0"
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                    torch_dtype=torch.float16)

# base = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_lora.safetensors" # Use the correct ckpt for your step setting!

SKIP_IF_EXISTS = True
USE_CONTROLNET=False

DENSEPOSE_NAME = "jschoormans/controlnet-densepose-sdxl"
controlnet = ControlNetModel.from_pretrained(
    DENSEPOSE_NAME,
    torch_dtype=torch.float16,
)



# Load model.
# You cant unload multiple loras -- is a problem for the replicate thing. But you could fuse the lightning and have only one lora every time I think...
# LORA
lorafolder = 'dress_dora/checkpoint-250'
tensors = load_file(f"{lorafolder}/pytorch_lora_weights.safetensors")

if USE_CONTROLNET:
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(base,
            vae=vae,torch_dtype=torch.float16, controlnet=controlnet,variant="fp16").to("cuda")
else:
    pipe= StableDiffusionXLPipeline.from_pretrained(base,
            vae=vae, torch_dtype=torch.float16, variant="fp16").to("cuda")


pipe.load_lora_weights(hf_hub_download(repo, ckpt), adapter_name="lightning")
pipe.load_lora_weights(tensors, adapter_name="clothing")

# load text
# handler = TokenEmbeddingsHandler(
#     [pipe.text_encoder, pipe.text_encoder_2],
#     [pipe.tokenizer, pipe.tokenizer_2]
# )
# handler.load_embeddings(f"{lorafolder}/dress_dora_emb.safetensors")

state_dict = load_file(f"{lorafolder}/dress_dora_emb.safetensors")
# load embeddings of text_encoder 1 (CLIP ViT-L/14)
pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
# load embeddings of text_encoder 2 (CLIP ViT-G/14)
pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

# # load params

prompt = "Barack Obama wearing a multicolored <s0><s1> dress, in New York, Brooklyn Bridge in the background"
# prompt = "A black man wearing a multicolored <s0><s1> dress, white sneakers, top hat, in Amsterdam, high detail beautiful, Vogue fashion photoshoot"
pipe.set_adapters(["lightning", "clothing"], adapter_weights=[1.0, 0.8])
# pipe.set_adapters(["clothing"], adapter_weights=[0.95])

# Fuses the LoRAs into the Unet
pipe.fuse_lora()

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Ensure using the same inference steps as the loaded model and CFG set to 0.



if USE_CONTROLNET:
    pipe(prompt, num_inference_steps=4, guidance_scale=0,
        control_image=load_image('brandon-dense.png'),
        image=load_image('brandon.png'),
        ).images[0].save('result.jpg')
else:
    pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0].save('result.jpg')
