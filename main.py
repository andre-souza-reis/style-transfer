# uvicorn main:app --reload --port 8080
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
from utils import process_image
from io import BytesIO
import cv2

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/uploadfile/")
async def create_upload_file(img: UploadFile, sty:UploadFile):
    
    image = np.asarray(bytearray(img.file.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image/255).astype(np.float32)

    style = np.asarray(bytearray(sty.file.read()), dtype="uint8")
    style = cv2.imdecode(style, cv2.IMREAD_COLOR)
    style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
    style = (style/255).astype(np.float32)

    processed_image = process_image(image, style)

    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
    res, im_png = cv2.imencode(".jpeg", processed_image)

    return StreamingResponse(BytesIO(im_png.tobytes()), media_type="image/jpeg")