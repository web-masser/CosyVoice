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
import time
import ssl  # 添加这行导入
from typing import Dict
from fastapi import FastAPI, UploadFile, Form, File, HTTPException, WebSocket
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
import ffmpeg
import torchaudio
import librosa

import random
from pathlib import Path
from filelock import FileLock

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:

        logging.info(f"进程 {os.getpid()} 开始初始化")
        
        with FileLock("gpu.lock"):
            if os.path.exists("gpu_ids.txt"):
                with open("gpu_ids.txt", "r") as f:
                    used_gpus = [int(line.strip()) for line in f.readlines()]
            else:
                used_gpus = []

            num_gpus = torch.cuda.device_count()
            worker_gpu = None  # 初始化 worker_gpu 变量

            # 检查已启动的 worker 数量
            num_workers = len(used_gpus)

            # 根据 worker 数量分配 GPU
            if num_workers == 0:
                worker_gpu = 0  # 第一个 worker 使用 GPU 0
            elif 1 <= num_workers <= 3:
                worker_gpu = 1  # 第二、三、四个 worker 使用 GPU 1
            else:
                raise Exception("No GPUs available for more than four workers")

            # 记录使用的 GPU
            with open("gpu_ids.txt", "a") as f:
                f.write(f"{worker_gpu}\n")

        global cosyvoice
        from cosyvoice.cli.cosyvoice import CosyVoice2
        torch.cuda.set_device(worker_gpu)
        app.state.thread_pool = ThreadPoolExecutor(max_workers=12)
        cosyvoice = CosyVoice2('D:/project/CosyVoice/pretrained_models/CosyVoice2-0.5B', 
                              load_jit=True, 
                              load_trt=False, 
                              fp16=False,
                              device_id=worker_gpu)  # 修改这里，直接传递设备对象
        
        yield
        
    except Exception as e:
        logging.error(f"lifespan 上下文中发生错误: {str(e)}", exc_info=True)
        raise
        
    finally:
        try:
            if 'cosyvoice' in globals():
                del cosyvoice
            torch.cuda.empty_cache()
            logging.info(f"进程 {os.getpid()} 清理完成")
        except Exception as e:
            logging.error(f"清理过程中发生错误: {str(e)}", exc_info=True)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

max_val = 0.8
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def convert_audio_to_16k(input_audio: io.BytesIO) -> bytes:
    # 使用 ffmpeg 转换音频到 16kHz
    out, _ = (
        ffmpeg
        .input('pipe:0')
        .filter('volume', '3dB')  # 降低音量增益
        .filter('atrim', duration=29)
        .output('pipe:1', 
                ar='16000',  # 采样率
                ac='1',      # 单声道
                format='wav',
                acodec='pcm_s16le',  # 使用16位PCM编码
                audio_bitrate='64k',  # 降低比特率
                compression_level='5'  # 压缩级别
        )
        .run(input=input_audio.read(), capture_stdout=True, capture_stderr=True)
    )
    return outd

@app.post("/inference/app-zero-save")
async def saveShot(fileName: str = Form(...), prompt_wav: UploadFile = File(...)):
    audio_data = await prompt_wav.read()
    converted_audio = convert_audio_to_16k(io.BytesIO(audio_data))
    with io.BytesIO(converted_audio) as f:
        prompt_speech_16k =  postprocess(load_wav(f, 16000))
    torchaudio.save(f"./py_data/{fileName}.wav", prompt_speech_16k, 16000, format="wav")
    torch.save(prompt_speech_16k, f"./py_data/{fileName}.pt", _use_new_zipfile_serialization=True)       
    return True           


# websocket-------------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_audio(self, client_id: str, audio_data: bytes):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_bytes(audio_data)


manager = ConnectionManager()

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    print("尝试建立 WebSocket 连接...")
    await websocket.accept()
    print("WebSocket 连接已接受")
    try:
        all_speech = []
        
        while True:                                      
            data = await websocket.receive_json()
            print(f"收到客户端消息: {data}")
            
            if data['type'] == 'generate':
                prompt_speech_16k = torch.load(f"./py_data/{data['file_name']}.pt")
                
                def run_inference():
                    if (data.get("language") not in [None, ""] or data.get("tone") not in [None, ""]):
                        instruct_text = "用"
                        
                        if data.get("tone") not in [None, ""]:
                            if data.get("language") not in [None, ""]:
                                instruct_text += data["tone"] + data["language"]
                            else:
                                instruct_text += data["tone"] + "的语气"
                        elif data.get("language") not in [None, ""]:
                            instruct_text += data["language"] + "的语气"
                            
                        instruct_text += "说"
                        print('inference_instruct2', data["tts_text"], instruct_text, prompt_speech_16k, data.get("stream", True), data.get("speed", 1.0))
                        return cosyvoice.inference_instruct2(
                            data["tts_text"],
                            instruct_text,
                            prompt_speech_16k,
                            stream=data.get("stream", True),
                            speed=data.get("speed", 1.0)
                        )
                    elif data.get("prompt_text", "") not in [None, ""]:                                                           
                        print('inference_zero_shot', data["tts_text"], data["prompt_text"], prompt_speech_16k, data.get("stream", True), data.get("speed", 1.0))
                        return cosyvoice.inference_zero_shot(
                            data["tts_text"],
                            data["prompt_text"],
                            prompt_speech_16k,
                            stream=data.get("stream", True),
                            speed=data.get("speed", 1.0)
                        )
                    else:
                        print('inference_cross_lingual', data["tts_text"], prompt_speech_16k, data.get("stream", True), data.get("speed", 1.0))
                        return cosyvoice.inference_cross_lingual(
                            data["tts_text"],
                            prompt_speech_16k,
                            stream=data.get("stream", True),
                            speed=data.get("speed", 1.0)
                        )
                
                print("开始执行推理...")
                future = app.state.thread_pool.submit(run_inference)
                model_output = await asyncio.wrap_future(future)
                print("推理完成，开始发送数据...")
                
                # 发送音频数据
                for i in model_output:
                    try:
                        all_speech.append(i['tts_speech'])
                        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                        # 确保数据大小合适
                        if len(tts_audio) > 0:
                            await websocket.send_bytes(tts_audio)
                            await asyncio.sleep(0.01)  # 添加延迟
                    except Exception as e:
                        print(f"发送音频数据时出错: {str(e)}")
                        break
                
                try:
                    # 发送完成信号
                    await websocket.send_json({"type": "complete", "status": "success"})
                    print("发送完成信号")
                    await asyncio.sleep(0.1)  # 等待客户端处理
                except Exception as e:
                    print(f"发送完成信号时出错: {str(e)}")
                
    except Exception as e:
        if str(e):  # 只有在有实际错误信息时才记录
            logging.error(f"WebSocket error: {str(e)}")
            print(f"发生错误: {str(e)}")
    finally:
        try:
            # 使用标准的关闭代码
            if not websocket.application_state == "closed":
                await websocket.close(code=1000, reason="Normal closure")
        except Exception as e:
            print(f"关闭连接时出错: {str(e)}")
        manager.disconnect(client_id)
        print(f"客户端 {client_id} 断开连接")
# -------------------------------------------------------------------------------  

def clear_gpu_ids():
    """清空 GPU ID 记录文件"""
    if os.path.exists("gpu_ids.txt"):
        os.remove("gpu_ids.txt")
    logging.info("GPU ID 记录已清空")

def release_resources():
    """释放所有资源，包括文件锁和 GPU ID 记录"""
    clear_gpu_ids()
    # 如果有其他需要释放的资源，也可以在这里添加
    logging.info("所有资源已释放")

if __name__ == '__main__':
    clear_gpu_ids()  # 在服务器启动前清空 GPU ID 文件
    try:
        logging.info("服务器开始启动")
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=6712,
            ssl_keyfile="./mznpy.com.key",
            ssl_certfile="./mznpy.com.pem",
            ws="websockets",
            workers=2
        )
    except Exception as e:
        logging.error(f"服务器启动失败: {str(e)}", exc_info=True)
    finally:
        release_resources()  # 确保在程序结束时释放所有资源
