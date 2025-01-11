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
import os
import sys
from concurrent import futures
import argparse
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import grpc
import torch
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from cosyvoice.utils.common import set_all_random_seed
import torchaudio
import tempfile
import io
import ffmpeg
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


class HeaderInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        # 获取请求地址
        method = handler_call_details.method
        print("Request Address:", method)

        # 获取请求头
        metadata = handler_call_details.invocation_metadata
        print("Request Headers:")
        for key, value in metadata:
            print(f"{key}: {value}")

        # 继续处理请求
        return continuation(handler_call_details)


class CosyVoiceServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    def __init__(self, args):
        try:
            self.cosyvoice = CosyVoice(args.model_dir)
        except Exception:
            try:
                self.cosyvoice = CosyVoice2(args.model_dir)
            except Exception:
                raise TypeError('no valid model_type!')
        logging.info('grpc service initialized')
        
    def convert_audio_to_16k(self, input_audio: bytes) -> bytes:
        # 使用 ffmpeg 转换音频到 16kHz
        out, _ = (
            ffmpeg
            .input('pipe:0')
            .filter('volume', '10dB')
            .filter('atrim', duration=28)
            .output('pipe:1', ar='16000', ac='1', format='wav')
            .run(input=input_audio, capture_stdout=True, capture_stderr=True)
        )
        return out

    def SaveShot(self, request, context):
        try:
            converted_audio = self.convert_audio_to_16k(request.prompt_audio)
            with io.BytesIO(converted_audio) as f:
                prompt_speech_16k = load_wav(f, 16000)
            
            # 保存音频文件
            torchaudio.save(f"./py_data/{request.file_name}.wav", 
                          prompt_speech_16k, 16000, format="wav")
            torch.save(prompt_speech_16k, f"./py_data/{request.file_name}.pt")
            
            return cosyvoice_pb2.SaveShotResponse(success=True)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.SaveShotResponse(success=False)

    def AppZeroOutput(self, request, context):
        try:
            set_all_random_seed(request.seed)
            prompt_speech_16k = torch.load(f"./py_data/{request.file_name}.pt")
            file_path = f"./output_data/{request.user_id}_single.wav"
            
            if os.path.exists(file_path):
                os.remove(file_path)
                
            model_output = self.cosyvoice.inference_zero_shot(
                request.tts_text,
                request.prompt_text,
                prompt_speech_16k,
                stream=request.stream,
                speed=request.speed
            )
            
            all_speech = []
            for i in model_output:
                response = cosyvoice_pb2.Response()
                response.tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                all_speech.append(i['tts_speech'])
                
                # 保存完整音频
                concatenated_speech = torch.cat(all_speech, dim=1)
                torchaudio.save(file_path, concatenated_speech, 22050, format="wav")
                
                yield response
                
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    def InferenceCrossLingual(self, request, context):
        try:
            set_all_random_seed(request.seed)
            prompt_speech_16k = torch.load(f"./py_data/{request.file_name}.pt")
            file_path = f"./output_data/{request.user_id}_single.wav"
            
            if os.path.exists(file_path):
                os.remove(file_path)
                
            model_output = self.cosyvoice.inference_cross_lingual(
                request.tts_text,
                prompt_speech_16k,
                stream=request.stream,
                speed=request.speed
            )
            
            all_speech = []
            for i in model_output:
                response = cosyvoice_pb2.Response()
                response.tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                all_speech.append(i['tts_speech'])
                
                concatenated_speech = torch.cat(all_speech, dim=1)
                torchaudio.save(file_path, concatenated_speech, 22050, format="wav")
                
                yield response
                
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

def main():
    # 加载证书和私钥
    # with open('./mznpy.com.pem', 'rb') as f:
    #     certificate_chain = f.read()
    # with open('./mznpy.com.key', 'rb') as f:
    #     private_key = f.read()

    # 创建 SSL 证书凭证
    # server_credentials = grpc.ssl_server_credentials(((private_key, certificate_chain),))

    # 创建 gRPC 服务器
    grpcServer = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_conc),
        maximum_concurrent_rpcs=args.max_conc,
        interceptors=[HeaderInterceptor()]
    )

    # 添加服务到服务器
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(CosyVoiceServiceImpl(args), grpcServer)

    # 使用不安全端口监听
    grpcServer.add_insecure_port('0.0.0.0:50051')  # 确保监听在所有接口上

    grpcServer.start()
    logging.info("server listening on 0.0.0.0:50051")
    grpcServer.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50051)
    parser.add_argument('--max_conc',
                        type=int,
                        default=10)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    main()
