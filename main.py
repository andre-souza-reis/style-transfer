# uvicorn main:app --reload --port 8080
from fastapi import FastAPI, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from utils import process_image
from io import BytesIO
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/files/")
async def create_file(img: bytes = File(...), sty: bytes = File(...)):
    
    image = np.asarray(bytearray(img))
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image/255).astype(np.float32)

    style = np.asarray(bytearray(sty))
    style = cv2.imdecode(style, cv2.IMREAD_COLOR)
    style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
    style = (style/255).astype(np.float32)

    processed_image = process_image(image, style)

    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
    res, im_png = cv2.imencode(".jpeg", processed_image)

    return StreamingResponse(BytesIO(im_png.tobytes()), media_type="image/jpeg")