from torch import autocast
import string    
import random
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

def image2image(file_path: str, 
    prompt,
    num_outputs=1,
    seed=42,
    guidance_scale=7.5
):
    # load the pipeline
    device = "cuda"
    generator = torch.Generator("cuda").manual_seed(seed)
    model_id_or_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token="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"
    )
    torch.cuda.empty_cache()

    pipe = pipe.to(device)
    init_image = Image.open(file_path).convert("RGB")
    init_image = init_image.resize((768, 512))
    with autocast("cuda"):
        images = pipe(
            prompt=[prompt]*num_outputs, 
            init_image=init_image, 
            strength=0.75, 
            guidance_scale=guidance_scale,
            generator=generator
        ).images
    print(len(images))
    paths = []
    for i in range(len(images)):
        ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
        images[i].save(f"outputs/img2img/{ran}.png")
        torch.cuda.empty_cache()
        paths.append(f"outputs/img2img/{ran}")
    print(paths)
    return paths
