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
from fastapi import FastAPI, UploadFile, Form, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
import ffmpeg
import torchaudio
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment 

import random
from pathlib import Path
from filelock import FileLock
from pydub import AudioSegment
from openunmix.predict import separate
from datetime import datetime
import re
import uuid

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
        global cosyvoice2
        from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice
        torch.cuda.set_device(worker_gpu)
        app.state.thread_pool = ThreadPoolExecutor(max_workers=12)
        cosyvoice = CosyVoice('D:/project/CosyVoice/pretrained_models/CosyVoice-300M', 
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
def postprocess(speech, top_db=120, hop_length=220, win_length=440):
    # 检查输入音频是否为空或太短
    if speech.numel() == 0 or speech.shape[-1] < win_length:
        print(f"警告: 输入音频太短或为空: {speech.shape}")
        # 返回一个小的默认音频片段而不是空音频
        return torch.zeros(1, int(cosyvoice.sample_rate * 0.5)) + 0.01
    
    # 检查音频是否全是静音或噪音
    if speech.abs().max() < 0.01:
        print(f"警告: 输入音频可能全是静音: max={speech.abs().max()}")
        return torch.zeros(1, int(cosyvoice.sample_rate * 0.5)) + 0.01
        
    try:
        # 保存原始音频以便回退
        original_speech = speech.clone()
        
        # 尝试trim处理
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        
        # 再次检查trim后的音频是否太短
        if speech.numel() == 0 or speech.shape[-1] < hop_length*2:
            print(f"警告: 音频trim后太短: {speech.shape}")
            speech = original_speech  # 回退到原始音频
            
        # 音量归一化
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
        elif speech.abs().max() < 0.1:  # 如果音量太小
            speech = speech / speech.abs().max() * 0.5  # 提高到适中音量
        # 添加尾部静音
        speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
        
        print(f"处理后音频信息: shape={speech.shape}, min={speech.min():.4f}, max={speech.max():.4f}, mean={speech.mean():.4f}")
        return speech
        
    except Exception as e:
        print(f"音频处理出错: {str(e)}")
        # 返回一个安全的默认音频而不是失败
        return torch.zeros(1, int(cosyvoice.sample_rate * 0.5)) + 0.01

def convert_audio_to_16k(input_audio: io.BytesIO) -> bytes:
    # 使用 ffmpeg 转换音频到 16kHz
    out, _ = (
        ffmpeg
        .input('pipe:0')
        .filter('volume', '7dB') 
        .filter('atrim', duration=20)
        .output('pipe:1', 
                ar='18000',
                format='wav',
        )
        .run(input=input_audio.read(), capture_stdout=True, capture_stderr=True)
    )
    return out

@app.post("/inference/app-zero-save")
async def saveShot(fileName: str = Form(...), prompt_wav: UploadFile = File(...)):
    audio_data = await prompt_wav.read()
    converted_audio = convert_audio_to_16k(io.BytesIO(audio_data))
    with io.BytesIO(converted_audio) as f:
        prompt_speech_16k =  postprocess(load_wav(f, 18000))
    torchaudio.save(f"./py_data/{fileName}.wav", prompt_speech_16k, 18000, format="wav")
    torch.save(prompt_speech_16k, f"./py_data/{fileName}.pt", _use_new_zipfile_serialization=True)       
    return True           

# 确保输出目录存在
output_dir = 'tq_person'
os.makedirs(output_dir, exist_ok=True)

def sanitize_filename(filename: str) -> str:
    """清理文件名，去掉不合法字符"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

@app.post("/inference/remove-background")
async def remove_background(audio_file: UploadFile = File(...), enable_compression: bool = Form(False)):
    # 清理文件名
    safe_filename = sanitize_filename(audio_file.filename)
    temp_path = f"temp_audio/{safe_filename}" 
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    with open(temp_path, "wb") as f:
        content = await audio_file.read() 
        f.write(content) 
    
    # 执行分离 
    output_path = separate_vocals(temp_path, enable_compression=enable_compression)

    # 返回分离结果
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="separated_vocals.wav" 
    )

def convert_to_wav(input_path: str, output_path: str) -> None:
    """将音频文件转换为 WAV 格式"""
    try:
        ffmpeg.input(input_path).output(output_path).run(overwrite_output=True)
    except Exception as e:
        raise RuntimeError(f"音频转换失败: {str(e)}")

def compress_audio(input_path: str, output_path: str, target_size_mb: float = 5.0) -> str:
    """压缩音频文件，确保文件小于指定大小"""
    target_size_bytes = target_size_mb * 1024 * 1024  # 转换为字节
    bitrate = 128  # 初始比特率

    while True:
        try:
            # 使用 ffmpeg 压缩音频
            ffmpeg.input(input_path).output(output_path, acodec='mp3', audio_bitrate=f'{bitrate}k').run(overwrite_output=True)
            
            # 检查文件大小
            if os.path.getsize(output_path) <= target_size_bytes:
                return output_path  # 返回压缩后的文件路径
            else:
                # 如果文件太大，降低比特率
                if bitrate > 32:  # 设置最低比特率限制
                    bitrate -= 16  # 每次降低 16k
                else:
                    raise RuntimeError("无法将音频压缩到小于 5MB，请尝试使用更短的音频文件或更低的质量设置。")
        except Exception as e:
            raise RuntimeError(f"音频压缩失败: {str(e)}")

def separate_vocals(input_path: str, output_dir: str = "output", enable_compression: bool = False) -> str:
    try:
        # 根据当前时间生成输出 WAV 文件的路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_output_path = f"temp_audio/converted_{timestamp}.wav"  # 使用时间戳命名输出文件
        
        # 确保临时目录存在
        os.makedirs(os.path.dirname(wav_output_path), exist_ok=True)

        # 先转换为 WAV 格式
        convert_to_wav(input_path, wav_output_path)

        # 使用 librosa 读取音频文件
        waveform, rate = librosa.load(wav_output_path, sr=None, mono=False)  # sr=None 保持原采样率
        audio_tensor = torch.from_numpy(waveform).float()  # 转换为 PyTorch 张量

        # 打印音频文件的详细信息
        print(f"音频文件采样率: {rate}, 音频形状: {audio_tensor.shape}")

        # 确保音频是单声道
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)  # 转换为单声道

        # 使用 Open-Unmix 的 separate 函数进行音轨分离
        estimates = separate(audio_tensor.unsqueeze(0), rate)  # 选择设备（'cpu' 或 'cuda'）

        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 保存人声音频到 tq_person 文件夹
        tq_person_dir = Path("tq_person")
        tq_person_dir.mkdir(exist_ok=True)  # 确保 tq_person 文件夹存在

        # 根据当前时间生成文件名
        vocal_path = tq_person_dir / f"vocals_{timestamp}.wav"  # 使用时间戳命名文件

        # 调整张量形状并保存
        vocals = estimates['vocals'].squeeze(0)  # 去掉多余的维度
        sf.write(str(vocal_path), vocals.numpy().T, rate)  # 保存人声

        # 如果需要压缩音频
        if enable_compression:
            compressed_output_path = f"temp_audio/compressed_{timestamp}.mp3"  # 压缩后的文件名
            compress_audio(vocal_path, compressed_output_path)  # 压缩音频
            return compressed_output_path  # 返回压缩后的文件路径

        return str(vocal_path)  # 返回分离后的人声路径
    except Exception as e:
        logging.error(f"音频处理失败: {str(e)}")  # 记录错误信息
        return ""  # 返回空字符串以指示失败

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

    def is_connected(self, client_id: str) -> bool:
        return client_id in self.active_connections and \
               self.active_connections[client_id].client_state != "disconnected"

    async def send_audio(self, client_id: str, audio_data: bytes):
        if self.is_connected(client_id):
            try:
                await self.active_connections[client_id].send_bytes(audio_data)
                return True
            except Exception as e:
                print(f"发送音频数据失败: {str(e)}")
                return False
        return False


manager = ConnectionManager()

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    print("尝试建立 WebSocket 连接...")
    connection_closed = False  # 添加标志来追踪连接状态
    
    try:
        await manager.connect(websocket, client_id)
        print(f"客户端 {client_id} WebSocket 连接已建立")
        
        all_speech = []
        
        while True:
            try:
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
                            return cosyvoice2.inference_instruct2(
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
                    
                    # 修改发送逻辑，增加连接状态检查
                    for i in model_output:
                        if websocket.client_state == "disconnected":
                            print("客户端已断开连接")
                            break
                            
                        try:
                            all_speech.append(i['tts_speech'])
                            tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                            if len(tts_audio) > 0:
                                await websocket.send_bytes(tts_audio)
                                await asyncio.sleep(0.01)
                        except Exception as e:
                            print(f"发送音频数据时出错: {str(e)}")
                            break
                    
                    # 只在连接仍然活跃时发送完成信号
                    if websocket.client_state != "disconnected":
                        try:
                            await websocket.send_json({"type": "complete", "status": "success"})
                            print("发送完成信号")
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            print(f"发送完成信号时出错: {str(e)}")
                            break
                
            except WebSocketDisconnect:
                print(f"客户端 {client_id} 断开连接")
                connection_closed = True  # 标记连接已关闭
                break
            except Exception as e:
                print(f"WebSocket处理过程中发生错误: {str(e)}")
                break
    except Exception as e:
        if str(e):
            logging.error(f"WebSocket error: {str(e)}")
            print(f"发生错误: {str(e)}")
    finally:
        try:
            manager.disconnect(client_id)
            print(f"客户端 {client_id} 连接已清理")
            
            # 只在连接未关闭且状态正常时尝试关闭
            if not connection_closed and \
               websocket.client_state != "disconnected" and \
               websocket.application_state != "closed":
                await websocket.close(code=1000, reason="Normal closure")
                
        except Exception as e:
            print(f"关闭连接时出错: {str(e)}")

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
            workers=1
        )
    except Exception as e:
        logging.error(f"服务器启动失败: {str(e)}", exc_info=True)
    finally:
        release_resources()  # 确保在程序结束时释放所有资源
