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
import soundfile as sf
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, Response, File, UploadFile, Form
from fastapi.responses import StreamingResponse
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
        app.cosyvoice = CosyVoice(model_dir)
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

class FileName(BaseModel):
    file_names: List[str]

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
    torchaudio.save(f'output.wav', output['tts_speech'], 22050, format="wav")
    torchaudio.save(buffer, output['tts_speech'], 22050, format="wav")
    buffer.seek(0)
    return Response(content=buffer.read(-1), media_type="audio/wav")

@app.post("/api/inference/app-zero-save")
async def saveShot(fileName: str = Form(...), audio: str = Form(...)):
    temp_file_path = base64_to_file(audio, '.wav')
    prompt_speech = load_wav(temp_file_path, 16000)
    prompt_audio = (prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
    prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
    prompt_speech_16k = prompt_speech_16k.float() / (2**15)
    file_name = f"./py_data/{fileName}.pt"
    torch.save(prompt_speech_16k, file_name)
    return True

# tts 输入的内容
# prompt 录制音频的内容
@app.post("/api/inference/app-zero-output")
async def outputShot(fileName: str = Form(...),tts: str = Form(...), prompt: str = Form(...)):
    start = time.process_time()
    file_name = f"./py_data/{fileName}.pt"
    prompt_speech_16k = torch.load(file_name)
    
    model_output = app.cosyvoice.inference_zero_shot(tts, prompt, prompt_speech_16k, stream=False)
    print(model_output)
    end = time.process_time()
    logging.info("infer time is {} seconds".format(end-start))
    result = next(model_output)
    return buildResponse(result)

@app.post("/api/inference/app-zero-multiple-output")
async def outputMultipleShot(fileName: str = Form(...),saveName: str = Form(...),tts: str = Form(...), prompt: str = Form(...)):
    if not fileName:
        raise ValueError("fileName cannot be empty")
    if not saveName:
        raise ValueError("saveName cannot be empty")
    file_name = f"./py_data/{fileName}.pt"
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"{file_name} does not exist")
    prompt_speech_16k = torch.load(file_name)
    
    model_output = app.cosyvoice.inference_zero_shot(tts, prompt, prompt_speech_16k, stream=False)
    if model_output is None:
        raise ValueError("model_output cannot be null")
    save_name = f"./temp_pt/{saveName}.pt"
    result = next(model_output)
    if result is None:
        raise ValueError("result cannot be null")
    torch.save(result, save_name)
    return buildResponse(result)

 #        
@app.post("/api/inference/app-zero-multiple-combine")
async def outputCombineShot(fileName = Form(...)):
    """
    This function takes in a comma-separated list of file names and combines the corresponding model outputs into a single audio file.
    """
    print("Received file names:", fileName)
    if not fileName:
        raise ValueError("fileName cannot be empty")
    if not isinstance(fileName, str):
        raise TypeError(f"fileName must be a string, but got {type(fileName)}")
    if not fileName.strip():
        raise ValueError("fileName cannot be empty")

    try:
        model_outputs = []
        for file_name in fileName.split(","):
            print("Processing file name:", file_name)
            # 对每个文件名执行某些操作
            file_url = f"./temp_pt/{file_name}.pt"
            if not os.path.exists(file_url):
                raise FileNotFoundError(f"{file_url} does not exist")
            model_output = torch.load(file_url)
            if model_output is None:
                raise ValueError(f"model output at {file_url} is null")
            if 'tts_speech' not in model_output:
                raise KeyError(f"model output at {file_url} does not contain key 'tts_speech'")
            model_outputs.append(model_output['tts_speech'])
        concatenated_tensor = torch.concat((model_outputs[0], model_outputs[1]), dim=1)

        audio_data = concatenated_tensor.cpu()
        buildResponse({'tts_speech': audio_data})
        
    except Exception as e:
        logging.error("Error occurred during outputCombineShot:", exc_info=True)
        raise e
    # return buildResponse(result)

if __name__ == "__main__":
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    uvicorn.run(app=app, host=ip_address, port=6070)