apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
uvicorn main:app --host 0.0.0.0 --port $PORT
