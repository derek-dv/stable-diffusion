from email.mime import image
from typing import Union
from src.text2img import text2image
from src.img2img import image2image
from src.inpaint import inpaint as inpaintImage
from fastapi import FastAPI, File, UploadFile, staticfiles

app = FastAPI()
app.mount("/images", staticfiles.StaticFiles(directory="outputs"), name="static")

def isImage(filename: str):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return True
    else:
        return False

@app.get("/")
def read_root():
    print("perfect")
    return {"Hello": "World"}


@app.get("/text2img")
def text2img(prompt: Union[str, None]):
    filename = text2image(prompt)
    return {"image": f"images/{filename}.png"}

@app.post("/img2img")
def img2img(prompt: str, file: UploadFile = File(...)):
    try:
        if not isImage(file.filename):
            raise Exception("File type uploaded not supported!")
        contents = file.file.read()
        file_name = "inputs/"+file.filename.replace(" ", "-")
        with open(file_name, 'wb') as f:
            f.write(contents)
        url = image2image(file_name, prompt)
        print(url)
        file.file.close()
        return {"image": f"images/{url}"}
        
    except Exception as err:
        file.file.close()
        return {"message": f"Error {err}"}       

@app.post("/inpaint")
def img2img(prompt: str, input: UploadFile = File(...), mask: UploadFile = File(...)):
    try:
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
        url = inpaintImage(input_file_name, mask_file_name, "rats on a  bench")
        mask.file.close()
        input.file.close()
        return {"image": f"images/{url}"}
        
    except Exception as err:
        return {"message": f"Error {err}"} 