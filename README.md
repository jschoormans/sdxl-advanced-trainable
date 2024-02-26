# Cog-SDXL

[![Replicate demo and cloud API](https://replicate.com/stability-ai/sdxl/badge)](https://replicate.com/stability-ai/sdxl)

This is an implementation of Stability AI's [SDXL](https://github.com/Stability-AI/generative-models) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

for prediction,

```bash
cog predict -i prompt="a photo of TOK"
```

```bash
cog train -i input_images=@example_datasets/__data.zip -i use_face_detection_instead=True
```

```bash
cog run -p 5000 python -m cog.server.http
```

## Update notes

**2023-08-17**
* ROI problem is fixed.
* Now BLIP caption_prefix does not interfere with BLIP captioner.


**2023-08-12**
* Input types are inferred from input name extensions, or from the `input_images_filetype` argument
* Preprocssing are now done with fp16, and if no mask is found, the model will use the whole image

**2023-08-11**
* Default to 768x768 resolution training
* Rank as argument now, default to 32
* Now uses Swin2SR `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` as default, and will upscale + downscale to 768x768



---
Install cog and requirements
    sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
    sudo chmod +x /usr/local/bin/cog
    sudo curl -o /usr/local/bin/pg
    et -L "https://github.com/replicate/pget/releases/download/v0.0.3/pget" && sudo ch
    mod +x /usr/local/bin/pget
pip install -r requirements.txt

Jasper:
I made a train script
python run_training.py 

And to run the inference do ???

sudo cog predict -i image=@brandon.png -i mask=@brandon-mask.png -i prompt="person with a man on its shirt"

I fucked up the predict script maybe but whatever

Now, I just need to know how to add the images and masks to the training script and test this. As of now, my images come out deep fried as fuck 

---
I have a spaces script to sync the lora checkpoints


Install AWS CLI (if not already installed):
    sudo apt update
    sudo apt install awscli
Configure AWS CLI for DigitalOcean Spaces:
    aws configure

AWS Access Key ID: Your DigitalOcean Spaces access key
AWS Secret Access Key: Your DigitalOcean Spaces secret key
Default region name: You can leave this blank or set it to a default region (e.g., us-east-1 for compatibility).
Default output format: You can leave this as json or blank.

The keys are: 
    LIOJFVFIBXKUZRYUZB23
    pS8vvVtlGIU1w9Tr7DALUSEU/kFotVeXJXZeMyD80vM

---

The inference server (while lora is training somewhere else -- syncing and creating figures):
    sudo ./download_from_s3.sh && python lightning.py && python create_grouped_images.py 