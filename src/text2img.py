from torch import autocast
from diffusers import StableDiffusionPipeline
import string    
import random, torch


def text2image(
    prompt="a photo of an astronaut riding a horse on mars", 
    num_outputs=1,
    width=512,
    height=512,
    seed=42,
    guidance_scale=7.5
):
    print("generating...")
    generator = torch.Generator("cuda").manual_seed(seed)
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        use_auth_token="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"
    )
    pipe = pipe.to("cuda")
    with autocast("cuda"):
        sample = pipe(
            prompt=[prompt]*num_outputs,
            width=width,
            height=height,
            generator=generator,
            guidance_scale=guidance_scale
        )#.sample[0]
        torch.cuda.empty_cache() 
        print(sample)
        images = sample["sample"]
        path = []
        for i in range(len(images)):
            ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
            filename = f"outputs/text2img/{ran}.png"
            images[i].save(filename)
            torch.cuda.empty_cache()
            path.append(f"outputs/text2img/{ran}")
        print(path)
        return path
