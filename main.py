from email.mime import image
from fileinput import filename
import io
import os
import cv2
from typing import Union
from src.text2img import text2image
from src.img2img import image2image
from src.inpaint import inpaint as inpaintImage
from fastapi import FastAPI, File, UploadFile, staticfiles, HTTPException
from starlette.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import tempfile
import torch

from src.tiles import Predictor

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def isImage(filename: str):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return True
    else:
        return False

@app.get("/image")
def get_image(image_path: str=""):
    try:
        img = cv2.imread(f"{image_path}.png")
        _, im_png = cv2.imencode(".png", img)
        os.remove(f"{image_path}.png")
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    except Exception as err:
        raise HTTPException(status_code=404, detail="File with given path does not exist")

@app.get("/text2img")
def text2img(prompt: Union[str, None], num_outputs: int = 1, guidance_scale: float = 7.5, height: int = 512, width: int = 512, seed: int = 41):
    torch.cuda.empty_cache()
    filename = text2image(prompt, num_outputs=num_outputs,
                          guidance_scale=guidance_scale, height=height, width=width, seed=seed)
    return {"image_paths": filename}


@app.post("/img2img")
def img2img(prompt: str, file: UploadFile = File(...), num_outputs: int = 1, guidance_scale: float = 7.5, seed: int = 41):
    try:
        torch.cuda.empty_cache()
        if not isImage(file.filename):
            raise Exception("File type uploaded not supported!")
        contents = file.file.read()
        file_name = "inputs/"+file.filename.replace(" ", "-")
        with open(file_name, 'wb') as f:
            f.write(contents)
        filename = image2image(
            file_name, prompt, guidance_scale=guidance_scale, num_outputs=num_outputs, seed=seed)
        return {"image_paths": filename}

    except Exception as err:
        file.file.close()
        raise HTTPException(status_code=403, detail=str(err))


@app.post("/inpaint")
def inpaint(prompt: str, input: UploadFile = File(...), mask: UploadFile = File(...), num_outputs: int = 1, guidance_scale: float = 7.5, height: int = 512, width: int = 512, seed: int = 41):
    try:
        torch.cuda.empty_cache()
        if not isImage(input.filename) or not isImage(mask.filename):
            raise Exception("File type uploaded not supported!")
        mask_contents = mask.file.read()
        mask_file_name = "inputs/"+mask.filename.replace(" ", "-")
        with open(mask_file_name, 'wb') as f:
            f.write(mask_contents)
        input_contents = input.file.read()
        input_file_name = "inputs/"+input.filename.replace(" ", "-")
        with open(input_file_name, 'wb') as f:
            f.write(input_contents)
        filename = inpaintImage(input_file_name, mask_file_name, prompt=prompt, seed=seed,
                                guidance_scale=guidance_scale, height=height, num_outputs=num_outputs, width=width)
        os.remove(input_file_name)
        os.remove(mask_file_name)
        return {"image_paths": filename}

    except Exception as err:
        raise HTTPException(status_code=403, detail=str(err))

@app.post("/tile_generate")
def tile(prompt: str, input: UploadFile = File(None), mask: UploadFile = File(None), num_outputs: int = 1, guidance_scale: float = 7.5, height: int = 512, width: int = 512, seed: int = 41):
    try:
        torch.cuda.empty_cache()
        print(input)
        if mask is None:
            mask_file_name = None
        else:
            if not isImage(mask.filename):
                raise Exception("File type uploaded not supported!")
            mask_contents = mask.file.read()
            mask_file_name = "inputs/"+mask.filename.replace(" ", "-")
            with open(mask_file_name, 'wb') as f:
                f.write(mask_contents)
        if input is None:
            input_file_name = None
        else:
            if not isImage(input.filename):
                raise Exception("File type uploaded not supported!")
            input_contents = input.file.read()
            input_file_name = "inputs/"+input.filename.replace(" ", "-")
            with open(input_file_name, 'wb') as f:
                f.write(input_contents)
        predict = Predictor()
        predict.setup()
        filename = predict.predict(init_image=input_file_name, mask=mask_file_name, prompt=prompt, seed=seed,
                                guidance_scale=guidance_scale, height=height, num_outputs=num_outputs, width=width)
        if (input_file_name): os.remove(input_file_name)
        if (mask_file_name): os.remove(mask_file_name)
        return {"image_paths": filename}

    except Exception as err:
        raise HTTPException(status_code=403, detail=str(err))
