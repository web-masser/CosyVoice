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
import torch
import numpy as np
import threading
import time
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out


class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.token_min_hop_len = 100
        self.token_max_hop_len = 200
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = 34
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.hift_cache_dict = {}

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device))
        self.llm.to(self.device).eval()
        self.llm.half()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model)
        self.flow.encoder = flow_encoder

    def load_onnx(self, flow_decoder_estimator_model):
        import onnxruntime
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
        del self.flow.decoder.estimator
        self.flow.decoder.estimator = onnxruntime.InferenceSession(flow_decoder_estimator_model, sess_options=option, providers=providers)

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        with self.llm_context:
            for i in self.llm.inference(text=text.to(self.device),
                                        text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_text=prompt_text.to(self.device),
                                        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                        prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                        embedding=llm_embedding.to(self.device).half(),
                                        sampling=25,
                                        max_token_text_ratio=30,
                                        min_token_text_ratio=3):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False):
        tts_mel = self.flow.inference(token=token.to(self.device),
                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                      prompt_token=prompt_token.to(self.device),
                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                      prompt_feat=prompt_feat.to(self.device),
                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                      embedding=embedding.to(self.device))
        # mel overlap fade in out
        if self.mel_overlap_dict[uuid] is not None:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
            self.hift_cache_dict[uuid] = {'source': tts_source[:, :, -self.source_cache_len:], 'mel': tts_mel[:, :, -self.mel_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
        return tts_speech

    def inference(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
              prompt_text=torch.zeros(1, 0, dtype=torch.int32),
              llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
              flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
              prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, **kwargs):
        """
        这是CosyVoice模型生成语音的主要入口函数。

        参数：
            text：要转换为语音的文本。
            flow_embedding：用于语音合成的流式嵌入。
            llm_embedding：用于语音合成的语言模型嵌入。默认为零。
            prompt_text：用于语音合成的提示文本。默认为零。
            llm_prompt_speech_token：用于语音合成的语言模型提示语音令牌。默认为零。
            flow_prompt_speech_token：用于语音合成的流式提示语音令牌。默认为零。
            prompt_speech_feat：用于语音合成的提示语音特征。默认为零。
            stream：是否流式输出。默认为False。

        返回：
            一个生成器，生成的语音字典。
        """
        # this_uuid用于跟踪此推理线程的变量
        this_uuid = str(uuid.uuid1())

        # 初始化此推理线程的变量
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.mel_overlap_dict[this_uuid], self.hift_cache_dict[this_uuid] = None, None
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()

        if stream is True:
            # 初始化流式输出的token_hop_len
            token_hop_len = self.token_min_hop_len

            while True:
                # 等待0.1秒后生成下一个语音块
                time.sleep(0.1)

                # 如果当前token长度大于token_hop_len，生成下一个语音块
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    # 获取下一个语音块的token
                    this_tts_speech_token = torch.concat(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len], dim=1)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]

                # 如果语言模型任务完成且没有更多token，退出循环
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break

            # 等待语言模型任务完成
            p.join()

            # 生成最后一个语音块
            this_tts_speech_token = torch.concat(self.tts_speech_token_dict[this_uuid], dim=1)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # 等待语言模型任务完成
            p.join()

            # 使用流式模型生成语音
            this_tts_speech_token = torch.concat(self.tts_speech_token_dict[this_uuid], dim=1)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}

        # 清理此推理线程的变量
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
