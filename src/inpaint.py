from torch import autocast
import torch
import PIL, random, string
from diffusers.pipelines import StableDiffusionInpaintPipeline

def get_image(url):
    return PIL.Image.open(url).convert("RGB")

def inpaint(
    input_image: str, 
    mask_image: str, 
    prompt, 
    num_outputs=1,
    width=512,
    height=512,
    seed=42,
    guidance_scale=7.5
):
    device = "cuda"
    model_id_or_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token="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"
    )

    pipe = pipe.to(device)
    generator = torch.Generator("cuda").manual_seed(seed)
    init_image = get_image(input_image).resize((512, 512))
    mask_image = get_image(mask_image).resize((512, 512))
    with autocast("cuda"):
        torch.cuda.empty_cache()
        images = pipe(
            prompt=[prompt]*num_outputs, 
            init_image=init_image, 
            mask_image=mask_image, 
            strength=0.75, 
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width
        ).images
    paths = []
    for i in range(len(images)):
        ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
        images[i].save(f"outputs/inpaint/{ran}.png")
        torch.cuda.empty_cache()
        paths.append(f"outputs/inpaint/{ran}")
    print(paths)
    return paths
