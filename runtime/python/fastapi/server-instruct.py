# Set inference model
# export MODEL_DIR=pretrained_models/CosyVoice-300M-Instruct
# For development
# fastapi dev --port 6006 fastapi_server.py
# For production deployment
# fastapi run --port 6006 fastapi_server.py
# conda activate cosyvoice
# python webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M

import os
import sys, socket
import io,time
import uvicorn
import base64
import tempfile
import wave
from datetime import datetime
from fastapi import FastAPI, Response, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware  #引入 CORS中间件模块
from contextlib import asynccontextmanager
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import numpy as np
import torch
import torchaudio
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class LaunchFailed(Exception):
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.getenv("MODEL_DIR", "D:/project/CosyVoice/pretrained_models/CosyVoice-300M")
    if model_dir:
        # logging.info("MODEL_DIR is {}", model_dir)
        app.cosyvoice = CosyVoice(model_dir)
        # sft usage
        # logging.info("Avaliable speakers {}", app.cosyvoice.list_avaliable_spks())
    else:
        raise LaunchFailed("MODEL_DIR environment must set")
    yield

app = FastAPI(lifespan=lifespan)

def base64_to_file(base64_string, file_extension):
    audio_data = base64.b64decode(base64_string)
    
     # 创建一个临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = temp_file.name
        
        # 创建一个WAV文件头
        with wave.open(temp_file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(44100)  # 采样率
            wav_file.writeframes(audio_data)
    
    return temp_file_path

#设置允许访问的域名
origins = ["*"]  #"*"，即为所有,也可以改为允许的特定ip。,"http://10.0.2.16"
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  #设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  #允许跨域的headers，可以用来鉴别来源等作用。

def buildResponse(output):
    buffer = io.BytesIO()
    torchaudio.save(buffer, output, 22050, format="wav")
    buffer.seek(0)
    return Response(content=buffer.read(-1), media_type="audio/wav")

@app.post("/api/inference/instruct")
async def instruct(tts: str = Form(), role: str = Form(), instruct: str = Form()):
    start = time.process_time()
    output = app.cosyvoice.inference_instruct(tts, role, instruct)
    end = time.process_time()
    logging.info("infer time is {} seconds", end-start)
    return buildResponse(output['tts_speech'])

if __name__ == "__main__":
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    print("vvvvvvvvvvvvvvvvvvvv-ip:" + ip_address)
    uvicorn.run(app=app, host=ip_address, port=6080)