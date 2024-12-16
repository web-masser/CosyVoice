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
from contextlib import asynccontextmanager
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
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio

# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks

thread_pool = ThreadPoolExecutor(max_workers=4)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 确保有两张 GPU
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError("需要两张 GPU 来运行服务")
    
    # 获取当前进程的 ID
    pid = os.getpid()
    
    # 根据进程 ID 分配 GPU
    gpu_id = 0 if pid % 2 == 0 else 1
    
    # 设置当前进程使用的 GPU
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)
    
    # 打印详细信息
    print(f"进程 {pid} 使用 GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    
    # 初始化模型等其他操作...
    from cosyvoice.cli.cosyvoice import CosyVoice
    global cosyvoice
    cosyvoice = CosyVoice('D:/project/CosyVoice/pretrained_models/CosyVoice-300M', True, True)
    
    app.state.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    yield
    
    # 关闭时执行
    # 清理代码
    app.state.thread_pool.shutdown()

app = FastAPI(lifespan=lifespan)

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
        .filter('volume', '10dB')
        .filter('atrim', duration=28)  # 只保留前30秒
        .output('pipe:1', ar='16000', ac='1', format='wav')
        .run(input=input_audio.read(), capture_stdout=True, capture_stderr=True)
    )
    return out

@app.post("/inference/app-zero-save")
async def saveShot(fileName: str = Form(...), prompt_wav: UploadFile = File(...)):
    audio_data = await prompt_wav.read()
    converted_audio = convert_audio_to_16k(io.BytesIO(audio_data))
    with io.BytesIO(converted_audio) as f:
        prompt_speech_16k = load_wav(f, 16000)
    torchaudio.save(f"./py_data/{fileName}.wav", prompt_speech_16k, 16000, format="wav")
    torch.save(prompt_speech_16k, f"./py_data/{fileName}.pt")       
    return True
        

def generate_data2(model_output, file_path):
    all_speech = []
    concatenated_speech = []
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        all_speech.append(i['tts_speech'])
        # 确保所有张量是2D并拼接
        concatenated_speech = torch.cat(all_speech, dim=1)  # 在时间维度拼接
        
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
    set_all_random_seed(int(input["seed"]))
    prompt_speech_16k = torch.load(f"./py_data/{input['file_name']}.pt")
    file_path = f"./output_data/{input['user_id']}_single.wav"
    if os.path.exists(file_path):
            os.remove(file_path)
    model_output = cosyvoice.inference_zero_shot(input["tts_text"], input["prompt_text"], prompt_speech_16k, stream=input["stream"], speed=input["speed"])
    return StreamingResponse(generate_data2(model_output, file_path), media_type="audio/wav")

@app.get("/audio/{type}/{user_id}")
async def get_audio(user_id: str, type: str):
    file_path = f"./output_data/{user_id}_{type}.wav"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"error": "File not found"}
    
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(input: dict):
    set_all_random_seed(int(input["seed"]))
    prompt_speech_16k = torch.load(f"./py_data/{input['file_name']}.pt")
    file_path = f"./output_data/{input['user_id']}_single.wav"
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # 使用线程池执行推理
    def run_inference():
        return cosyvoice.inference_cross_lingual(
            input["tts_text"], 
            prompt_speech_16k, 
            stream=input["stream"], 
            speed=input["speed"]
        )
    
    # 在线程池中执行推理，并使用 asyncio.wrap_future 包装
    future = app.state.thread_pool.submit(run_inference)
    model_output = await asyncio.wrap_future(future)
    
    # 保持原有的流式响应
    return StreamingResponse(
        generate_data2(model_output, file_path), 
        media_type="audio/wav"
    )
    
# @app.post("/translate")
# async def inference_cross_lingual(input: dict):
#     input_sequence = input['text']
#     pipeline_ins = pipeline(task=Tasks.translation, model="damo/nlp_csanmt_translation_zh2en")
#     outputs = pipeline_ins(input=input_sequence)
#     return outputs

@app.post("/inference_instruct")
async def inference_instruct(input: dict):
    set_all_random_seed(int(input["seed"]))
    file_path = f"./output_data/{input['user_id']}_single.wav"
    if os.path.exists(file_path):
            os.remove(file_path)
    logging.info(input["natural_text"])
    if input["natural_text"].endswith(('。', ',', '，', '、', '.', '?', '!')):
        input["natural_text"] = input["natural_text"][:-1]
    
    model_output = cosyvoice_instruct.inference_instruct(input["tts_text"], input["speak"], input["natural_text"],stream=input["stream"], speed=input["speed"])
    return StreamingResponse(generate_data2(model_output, file_path), media_type="audio/wav")


if __name__ == '__main__':
    uvicorn.run(app="server:app", host="127.0.0.1", port=6070, workers=2)
    # uvicorn.run(app=app, host="127.0.0.1", port=6712)
