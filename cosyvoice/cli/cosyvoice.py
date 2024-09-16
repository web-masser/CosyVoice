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
import time
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.utils.file_utils import logging

class CosyVoice:

    def __init__(self, model_dir, load_jit=True):
        logging.info("model_dir2222:{}".format(model_dir))
        instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        logging.info("os.path.exists(model_dir):{}".format(os.path.exists(model_dir)))
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          instruct,
                                          configs['allowed_special'])
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'])
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.fp16.zip'.format(model_dir),
                                    '{}/llm.llm.fp16.zip'.format(model_dir))
        del configs

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id, stream=False):
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text1 {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False):
        # 这里做一个接口  --  做成等待式 ----  例如 ：  正在切分句子
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text2 {}'.format(i))
            logging.info('model_input {}'.format(model_input))
            
            # ------------------------------------------------------------------
            
            # 这里就变成 输出，  等待示例： 正在返回音频
            for model_output in self.model.inference(**model_input, stream=stream):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False):
        if self.frontend.instruct is True:
            raise ValueError('{} do not support cross_lingual inference'.format(self.model_dir))
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text3 {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False):
        """
        执行指令推理。

        该函数接受四个参数：要合成的文本、说话人ID、指令文本和是否流式输出。
        它首先检查模型是否支持指令推理，如果不支持，则抛出异常。
        然后，它对指令文本进行归一化处理，分割文本为子词并将其转换为整数。
        接下来，它循环处理要合成的文本，准备模型的输入数据，调用模型进行推理，并输出结果。

        参数：
            tts_text (str): 要合成的文本。
            spk_id (str): 说话人ID。
            instruct_text (str): 指令文本。
            stream (bool): 是否流式输出。

        输出：
            dict: 输出语音和其他信息。
        """

        # 检查模型是否支持指令推理
        if not self.frontend.instruct:
            raise ValueError('{} 不支持指令推理'.format(self.model_dir))

        # 归一化指令文本
        instruct_text = self.frontend.text_normalize(instruct_text, split=False)

        # 循环处理要合成的文本
        for i in self.frontend.text_normalize(tts_text, split=True):
            # 准备模型的输入数据
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)

            # 开始计时
            start_time = time.time()

            # 输出进度信息
            logging.info('合成文本 {}'.format(i))

            # 循环输出语音
            for model_output in self.model.inference(**model_input, stream=stream):
                # 计算输出语音的长度
                speech_len = model_output['tts_speech'].shape[1] / 22050

                # 输出进度信息
                logging.info('输出语音长度 {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))

                # 输出语音
                yield model_output

                # 重置计时器
                start_time = time.time()
