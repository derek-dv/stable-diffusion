import os
from typing import Optional, List
import numpy as np
import torch, string, random
import torch.nn as nn
from torch import autocast
from diffusers import PNDMScheduler, LMSDiscreteScheduler
from PIL import Image
from src.tiles_func import StableDiffusionImg2ImgPipeline


def preprocess_init_image(image: Image, width: int, height: int):
    image = image.resize((width, height), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask: Image, width: int, height: int):
    mask = mask.convert("L")
    mask = mask.resize((width // 8, height // 8), resample=Image.LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = torch.from_numpy(mask)
    return mask

def patch_conv(**patch):
    cls = torch.nn.Conv2d
    init = cls.__init__
    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, **patch)
    cls.__init__ = __init__

patch_conv(padding_mode='circular')

MODEL_CACHE = "diffusers-cache"


class Predictor:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        scheduler = PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token="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"
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        init_image = None,
        mask = None,
        prompt_strength: float = 0.8,
        num_outputs: int = 1,
        num_inference_steps: int=50,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ):
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width == height == 1024:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        if init_image:
            init_image = Image.open(init_image).convert("RGB")
            init_image = preprocess_init_image(init_image, width, height).to("cuda")

            # use PNDM with init images
            scheduler = PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )
        else:
            # use LMS without init images
            scheduler = LMSDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )

        self.pipe.scheduler = scheduler

        if mask:
            mask = Image.open(mask).convert("RGB")
            mask = preprocess_mask(mask, width, height).to("cuda")

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=[prompt] * num_outputs,# num_outputs if prompt is not None else None,
            init_image=init_image,
            mask=mask,
            width=width,
            height=height,
            prompt_strength=prompt_strength,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )
        torch.cuda.empty_cache()
        print(output)
        output_paths = []
        for _, sample in enumerate(output["sample"]):
            ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
            output_path = f"outputs/tiles/{ran}.png"
            sample.save(output_path)
            print(sample)
            output_paths.append(f"outputs/tiles/{ran}")
        torch.cuda.empty_cache()
        
        return output_paths

