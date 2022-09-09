from torch import autocast
import string    
import random
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

# load the pipeline
device = "cuda"
model_id_or_path = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token="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"
)

pipe = pipe.to(device)

def image2image(file_path: str, prompt: str="A fantasy landscape, trending on artstation"):
    init_image = Image.open(file_path).convert("RGB")
    init_image = init_image.resize((768, 512))
    with autocast("cuda"):
        images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5).images
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
    images[0].save(f"outputs/img2img/{ran}.png")
    return f"img2img/{ran}.png"