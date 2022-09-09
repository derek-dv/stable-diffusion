from torch import autocast
import torch, os
import PIL, random, string

from diffusers import StableDiffusionInpaintPipeline

def get_image(url):
    return PIL.Image.open(url).convert("RGB")

device = "cuda"
model_id_or_path = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id_or_path,
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token="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"
)

pipe = pipe.to(device)

def inpaint(input_image: str, mask_image: str, prompt):
    prompt = "a cat sitting on a bench"
    init_image = get_image(input_image).resize((512, 512))
    mask_image = get_image(mask_image).resize((512, 512))
    with autocast("cuda"):
        images = pipe(prompt=prompt, init_image=init_image, mask_image=mask_image, strength=0.75).images
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
    os.mkdir("outputs/inpaint")
    images[0].save(f"outputs/inpaint/{ran}.png")
    return f"inpaint/{ran}.png"