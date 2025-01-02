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
import time
import json
import ssl  # 添加这行导入
# import aioredis
# from aioredis import Redis
from typing import Dict
from fastapi import FastAPI, UploadFile, Form, File, HTTPException, WebSocket
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio

import tensorrt as trt
import torch

thread_pool = ThreadPoolExecutor(max_workers=12)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

print(sys.version)
print(ssl.OPENSSL_VERSION)

# 在文件顶部添加全局配置
NUM_WORKERS = torch.cuda.device_count()  # 根据 GPU 数量设置 worker 数
print(f"检测到 {NUM_WORKERS} 个 GPU，将启动相同数量的 workers")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 检查可用的 GPU
    gpu_count = torch.cuda.device_count()
    
    # 获取当前进程的 ID 并等待一小段时间
    # 这样可以让进程按顺序启动并分配到不同的 GPU
    pid = os.getpid()
    await asyncio.sleep(0.1 * (pid % gpu_count))  # 错开启动时间
    
    worker_count_file = ".worker_count"
    worker_id = 0

    cleanup_task = None
    
    async def cleanup_buffers():
        while True:
            try:
                output_dir = "./output_data"
                current_time = time.time()
                for file in os.listdir(output_dir):
                    if "_buffer" in file:
                        file_path = os.path.join(output_dir, file)
                        # 删除超过5分钟的缓冲文件
                        if current_time - os.path.getmtime(file_path) > 300:
                            safe_remove(file_path)
            except Exception as e:
                logging.error(f"Error in cleanup: {str(e)}")
            await asyncio.sleep(300)  
    
    try:
        if os.path.exists(worker_count_file):
            with open(worker_count_file, 'r') as f:
                worker_id = int(f.read().strip()) + 1
        with open(worker_count_file, 'w') as f:
            f.write(str(worker_id))
    except Exception as e:
        print(f"警告：worker计数出错 - {e}")
        worker_id = pid % gpu_count
    
    # 分配 GPU（按进程启动顺序）
    gpu_id = worker_id % gpu_count
    print(f"进程 {pid} 使用 GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")

    # 在初始化模型之前清理 GPU 缓存
    torch.cuda.empty_cache()
    
    # 确保当前进程使用指定的 GPU
    torch.cuda.set_device(gpu_id)
    
    
    try:
        # 启动清理任务
        cleanup_task = asyncio.create_task(cleanup_buffers())
        
        # 初始化其他资源...
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
        global cosyvoice
        cosyvoice = CosyVoice2('D:/project/CosyVoice/pretrained_models/CosyVoice2-0.5B', 
                              load_jit=True, load_onnx=True, load_trt=False, deviceId=1)
        app.state.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        yield
        
    finally:
        # 清理资源
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 清理其他资源
        app.state.thread_pool.shutdown()
        
        # 最后一次清理所有缓冲文件
        try:
            output_dir = "./output_data"
            for file in os.listdir(output_dir):
                if "_buffer" in file:
                    file_path = os.path.join(output_dir, file)
                    safe_remove(file_path)
        except Exception as e:
            logging.error(f"Error in final cleanup: {str(e)}")

    # 初始化模型等其他操作...
    # from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    # global cosyvoice
    # # cosyvoice = CosyVoice('D:/project/CosyVoice/pretrained_models/CosyVoice-300M', True, True)
    # cosyvoice = CosyVoice2('D:/project/CosyVoice/pretrained_models/CosyVoice2-0.5B', load_jit=True, load_onnx=True, load_trt=False, deviceId=1)
    # app.state.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    # yield
    
    # # 关闭时执行
    # # 清理代码
    # app.state.thread_pool.shutdown()

app = FastAPI(lifespan=lifespan)

# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)
        
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

def get_buffer_paths(file_path):
    """获取双缓冲文件路径"""
    base, ext = os.path.splitext(file_path)
    return f"{base}_buffer1{ext}", f"{base}_buffer2{ext}"

def safe_remove(file_path):
    """安全删除文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        return True
    except Exception as e:
        logging.error(f"Error removing file {file_path}: {str(e)}")
        return False

def generate_data2(model_output, file_path):
    all_speech = []
    concatenated_speech = []
    
    # 获取双缓冲文件路径
    buffer1, buffer2 = get_buffer_paths(file_path)
    
    # 决定使用哪个缓冲文件
    if os.path.exists(buffer1):
        current_buffer = buffer2
        old_buffer = buffer1
    else:
        current_buffer = buffer1
        old_buffer = buffer2
        
    try:
        for i in model_output:
            tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
            all_speech.append(i['tts_speech'])
            concatenated_speech = torch.cat(all_speech, dim=1)
            
            try:
                # 保存到当前缓冲文件
                torchaudio.save(current_buffer, concatenated_speech, 22050, format="wav")
                
                # 等待文件写入完成
                time.sleep(0.1)
                
                # 使用 shutil.copy2 替代硬链接
                import shutil
                try:
                    shutil.copy2(current_buffer, file_path)
                except Exception as e:
                    logging.error(f"Error copying file: {str(e)}")
                
                # 现在旧的缓冲文件已经不再被使用，可以安全删除
                if os.path.exists(old_buffer) and old_buffer != current_buffer:
                    safe_remove(old_buffer)
                
            except Exception as e:
                logging.error(f"Error in file operation: {str(e)}")
                
            yield tts_audio
            
    except Exception as e:
        logging.error(f"Error in generate_data2: {str(e)}")
    finally:
        # 确保清理临时文件
        try:
            safe_remove(current_buffer)
            safe_remove(old_buffer)
        except:
            pass

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
    if not os.path.exists(file_path):
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

@app.websocket("/test")
async def test_websocket(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("连接成功")
    await websocket.close()


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
            await websocket.close(code=1000, reason="Normal closure")
        except Exception as e:
            print(f"关闭连接时出错: {str(e)}")
        manager.disconnect(client_id)
        print(f"客户端 {client_id} 断开连接")
# -------------------------------------------------------------------------------  

def create_ssl_context():
    # 创建 SSL 上下文
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    # 基本配置
    ssl_context.verify_mode = ssl.CERT_NONE
    ssl_context.check_hostname = False
    
    # 设置协议版本
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
    ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3
    
    # 设置加密套件 (特别是 iOS 支持的)
    ssl_context.set_ciphers(':'.join([
        'ECDHE-ECDSA-AES128-GCM-SHA256',
        'ECDHE-RSA-AES128-GCM-SHA256',
        'ECDHE-ECDSA-AES256-GCM-SHA384',
        'ECDHE-RSA-AES256-GCM-SHA384',
        'DHE-RSA-AES128-GCM-SHA256',
        'DHE-RSA-AES256-GCM-SHA384',
        'TLS_AES_128_GCM_SHA256',  # TLS 1.3
        'TLS_AES_256_GCM_SHA384'   # TLS 1.3
    ]))
    
    return ssl_context

if __name__ == '__main__':
    try:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.verify_mode = ssl.CERT_NONE
        context.check_hostname = False
        
        # macOS 偏好 TLS 1.2
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_2  # 限制为 TLS 1.2
        
        # 使用 macOS 原生支持的加密套件
        context.set_ciphers('ECDHE-RSA-AES128-GCM-SHA256')  # 只用一个最基本的
        
        print("SSL 配置完成，准备启动服务器...")
        
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=6712,
            ssl_keyfile="./mznpy.com.key",
            ssl_certfile="./mznpy.com.pem",
            ssl_version=ssl.PROTOCOL_TLSv1_2,
            log_level="debug",
            ws="websockets",
            http="h11",
            ws_max_size=1024*1024*10,
            ws_ping_interval=20,
            ws_ping_timeout=30,
            ws_per_message_deflate=False
        )
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
