from torch import autocast
from diffusers import StableDiffusionPipeline
import string    
import random

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    use_auth_token="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"
)
pipe = pipe.to("cuda")

def text2image(prompt="a photo of an astronaut riding a horse on mars"):
    print("generating...")
    with autocast("cuda"):
        sample = pipe(prompt)#.sample[0] 
        image = sample["sample"][0]
        ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
        filename = f"outputs/text2img/{ran}.png"
        image.save(filename)
    return f"text2img/{ran}.png"
text2image()