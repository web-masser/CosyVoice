# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from io import BytesIO
import io
import os
import sys
import logging
import torch
import ffmpeg
import torchaudio
import tempfile
import uuid
import aioredis
from aioredis import Redis
from typing import Dict
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File, WebSocket, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()

# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])
        
def convert_audio_to_16k(input_audio: io.BytesIO) -> bytes:
    # 使用 ffmpeg 转换音频到 16kHz
    out, _ = (
        ffmpeg
        .input('pipe:0')
        .output('pipe:1', ar='16000', ac='1', format='wav')
        .run(input=input_audio.read(), capture_stdout=True, capture_stderr=True)
    )
    return out

@app.on_event("startup")
async def startup_event():
    global cosyvoice
    cosyvoice = CosyVoice('iic/CosyVoice-300M')
    global redis
    redis = aioredis.from_url("redis://localhost")
    
@app.on_event("shutdown")
async def shutdown():
    # 在应用关闭时执行的代码
    print("Closing Redis connection...")
    await redis.close()

@app.post("/inference/app-zero-save")
async def saveShot(fileName: str = Form(...), prompt_wav: UploadFile = File(...)):
    audio_data = await prompt_wav.read()
    converted_audio = convert_audio_to_16k(io.BytesIO(audio_data))
    with io.BytesIO(converted_audio) as f:
        prompt_speech_16k = load_wav(f, 16000)
        torch.save(prompt_speech_16k, f"./py_data/{fileName}.pt")       
    return True
        

def generate_data2(model_output, file_path):
    all_speech = []
    concatenated_speech = []
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        all_speech.append(i['tts_speech'])
        # 确保所有张量是2D并拼接
        concatenated_speech = torch.cat(all_speech, dim=1)  # 在时间维度上拼接
        
        with tempfile.NamedTemporaryFile(delete=False, dir="./output_data", suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
            # 保存音频到临时文件
            torchaudio.save(temp_file_path, concatenated_speech, 22050, format="wav")
            
        if os.path.exists(file_path):
            os.remove(file_path)
        # 保存到文件
        os.rename(temp_file_path, file_path)
        yield tts_audio        
        
@app.post("/app-zero-output")
async def inference_zero_shot(input: dict):
    prompt_speech_16k = torch.load(f"./py_data/{input['file_name']}.pt")
    file_path = f"./output_data/{input['user_id']}_single.wav"
    if os.path.exists(file_path):
            os.remove(file_path)
    model_output = cosyvoice.inference_zero_shot(input["tts_text"], input["prompt_text"], prompt_speech_16k, stream=True, speed=input["speed"])
    return StreamingResponse(generate_data2(model_output, file_path), media_type="audio/wav")    

@app.get("/audio/{type}/{user_id}")
async def get_audio(user_id: str, type: str):
    file_path = f"./output_data/{user_id}_{type}.wav"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"error": "File not found"}

@app.get("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data2(model_output))


@app.get("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data2(model_output))


if __name__ == '__main__':
    # uvicorn.run(app="server:app", host="192.168.66.108", port=6070, workers=2)
    uvicorn.run(app=app, host="192.168.66.108", port=6070)
