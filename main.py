# uvicorn main:app --reload --port 8080
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
from PIL import Image
from utils import process_image
from io import BytesIO
import cv2

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id : int):
    return {"item_id": item_id}

@app.post("/files/")
async def create_file(file: bytes = File(...)):
    print(type(file))
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(img: UploadFile):
    pil_img = Image.open(img.file)
    image = np.array(pil_img, dtype=np.float32)/255

    processed_image = process_image(image)*255

    res, im_png = cv2.imencode(".jpeg", processed_image)

    return StreamingResponse(BytesIO(im_png.tobytes()), media_type="image/jpeg")