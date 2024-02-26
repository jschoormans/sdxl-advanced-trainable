import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, StableDiffusionXLControlNetPipeline, ControlNetModel
from huggingface_hub import hf_hub_download
from dataset_and_utils import TokenEmbeddingsHandler
from safetensors.torch import load_file
import json
import os
from diffusers.utils import load_image


base = "stabilityai/stable-diffusion-xl-base-1.0"
# base = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_lora.safetensors" # Use the correct ckpt for your step setting!

SKIP_IF_EXISTS = True

DENSEPOSE_NAME = "jschoormans/controlnet-densepose-sdxl"
controlnet = ControlNetModel.from_pretrained(
    DENSEPOSE_NAME,
    torch_dtype=torch.float16,
)




OUTPUTFOLDER = 'multi_lora_training_dress'
OUTPUTFOLDER = 'multi_lora_training_dress_run3'


input_images_list = ["input.zip", "input_albu.zip"]
RESOLUTIONS = [512, 768]
lrs = [1e-5, 1e-4, 1e-3, 5e-5, 5e-6] 

steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]


for USE_CONTROLNET in [True, False]:
    for RESOLUTION in RESOLUTIONS:
        for input_images in input_images_list:
            for lr in lrs:
                for step in steps:
                    try:
                        
                        SAVENAME = "run3/output_" + str(RESOLUTION) + "_" + str(lr) + "_" + input_images.split(".")[0] + "-" + str(step) + "CONTROL" + str(USE_CONTROLNET) + "_unet1e-6.png"
                        
                        
                        # SKIPS IF EXISTS (ONLY IF SKIP_IF_EXISTS IS TRUE)
                        if SKIP_IF_EXISTS and os.path.exists(OUTPUTFOLDER + os.path.sep + SAVENAME):
                            print("Skipping", step, input_images, RESOLUTION, lr)
                            continue


                        # Load model.
                        # You cant unload multiple loras -- is a problem for the replicate thing. But you could fuse the lightning and have only one lora every time I think...
                        # LORA
                        LORA_DIR = "training_out_" + str(RESOLUTION) + "_" + str(lr) + "_" + input_images.split(".")[0]
                        lorafolder = 'trainings/' + LORA_DIR
                        tensors = load_file(f"{lorafolder}/checkpoints/unet/checkpoint-{step}.lora.safetensors")
                        
                        
                        
                        if USE_CONTROLNET:
                            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(base, torch_dtype=torch.float16, 
                                                                                            controlnet=controlnet,variant="fp16").to("cuda")
                        else:
                            pipe= StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16").to("cuda")


                        pipe.load_lora_weights(hf_hub_download(repo, ckpt), adapter_name="lightning")
                        pipe.load_lora_weights(tensors, adapter_name="clothing")

                        # load text
                        handler = TokenEmbeddingsHandler(
                            [pipe.text_encoder, pipe.text_encoder_2],
                            [pipe.tokenizer, pipe.tokenizer_2]
                        )
                        handler.load_embeddings(f"{lorafolder}/checkpoints/embeddings/checkpoint-{step}.pti")

                        # load params
                        with open(f"{lorafolder}/special_params.json", "r") as f:
                            params = json.load(f)
                        token_map = params
                        tuned_model = True

                        print(token_map)

                        prompt = "A woman wearing a multicolored TOK dress"
                        for k, v in token_map.items():
                            prompt = prompt.replace(k, v)
                            print(f"Prompt: {prompt}")

                        pipe.set_adapters(["lightning", "clothing"], adapter_weights=[1.0, 1.0])
                        # Fuses the LoRAs into the Unet
                        pipe.fuse_lora()

                        # Ensure sampler uses "trailing" timesteps.
                        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

                        # Ensure using the same inference steps as the loaded model and CFG set to 0.
                        
                        
                        
                        if USE_CONTROLNET:
                            pipe(prompt, num_inference_steps=4, guidance_scale=0,
                                control_image=load_image('brandon-dense.png'),
                                image=load_image('brandon.png'),
                                ).images[0].save(OUTPUTFOLDER + os.path.sep + SAVENAME)
                        else:
                            pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0].save(OUTPUTFOLDER + os.path.sep + SAVENAME)
                        
                        #
                    except Exception as e:
                        print(e)
                        print("Failed to generate image for", step, input_images, RESOLUTION, lr)
                        continue
                    
                    # unload pipe
                    pipe = None