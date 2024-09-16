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
from functools import partial
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import os
import re
import inflect
try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    use_ttsfrd = False
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph


from cosyvoice.utils.file_utils import logging


class CosyVoiceFrontEnd:

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 spk2info: str = '',
                 instruct: bool = False,
                 allowed_special: str = 'all'):
        """
        初始化CosyVoice前端。

        参数：
            get_tokenizer：一个返回tokenizer实例的可调用函数。
            feat_extractor：一个返回特征提取器实例的可调用函数。
            campplus_model：campplus模型的路径。
            speech_tokenizer_model：语音tokenizer模型的路径。
            spk2info：说话人信息文件的路径（可选）。
            instruct：是否使用指令推理（默认为False）。
            allowed_special：允许的特殊token（默认为'all'）。

        注意：
            * 如果不提供说话人信息文件，前端将使用默认的说话人信息。
            * 允许的特殊token是不会被tokenizer删除的token。默认情况下，所有特殊token都是允许的。
        """
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the campplus model
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])

        # Load the speech tokenizer model
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option, providers=["CUDAExecutionProvider"if torch.cuda.is_available() else "CPUExecutionProvider"])

        # Load the speaker info file
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)

        # Set the instruct flag
        self.instruct = instruct

        # Set the allowed special tokens
        self.allowed_special = allowed_special

        # Initialize the inflect parser
        self.inflect_parser = inflect.engine()

        # Initialize the ttsfrd engine
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            # Initialize the ttsfrd engine
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/../../pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, 'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyin')
            self.frd.enable_pinyin_mix(True)
            self.frd.set_breakmodel_index(1)
        else:
            # Initialize the ZhNormalizer and EnNormalizer
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False)
            self.en_tn_model = EnNormalizer()

    def _extract_text_token(self, text):
        """
        从给定的文本中提取文本标记（token）.


        参数：
            text (str): 输入文本.


        返回：
            tuple: 一个包含提取的文本标记和其长度的元组.
        """
        # 使用提供的分词器（tokenizer）对文本进行分词
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)


        # 将分词后的文本转换为张量并移动到设备（device）
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)


        # 获取文本标记的长度
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)


        # 返回提取的文本标记和其长度
        return text_token, text_token_len

    def _extract_speech_token(self, speech):
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None, {self.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                                                self.speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None, {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True):
        text = text.strip()
        if contains_chinese(text):
            if self.use_ttsfrd:
                text = self.frd.get_frd_extra_info(text, 'input')
            else:
                text = self.zh_tn_model.normalize(text)
            logging.info('text {}'.format(text))
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "、")
            text = text.replace(" - ", "，")
            text = text.replace("！", "，")  # 将文本中的破折号替换为中文逗号
            text = text.replace("!", "，")  # 将文本中的破折号替换为中文逗号
            text = remove_bracket(text)
            text = re.sub(r'[，,、]+$', '。', text)
            texts = [i for i in split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                                token_min_n=60, merge_len=20,
                                                comma_split=False)]
        else:
            if self.use_ttsfrd:
                text = self.frd.get_frd_extra_info(text, 'input')
            else:
                text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
            texts = [i for i in split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                                token_min_n=60, merge_len=20,
                                                comma_split=False)]
        if split is False:
            return text
        return texts

    def frontend_sft(self, tts_text, spk_id):
        """
        prepare input for sft mode

        1. first, we extract text token and text token length from tts_text
        2. then, we get speaker embedding from spk2info
        3. finally, we create model_input with text, text_len, llm_embedding, and flow_embedding
        """
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        # 张量？
        embedding = self.spk2info[spk_id]['embedding'] 
        model_input = {
            'text': tts_text_token,  # text token
            'text_len': tts_text_token_len,  # text token length
            'llm_embedding': embedding,  # speaker embedding for llm
            'flow_embedding': embedding  # speaker embedding for flow
        }
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                       'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                       'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                       'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                       'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k):
        model_input = self.frontend_zero_shot(tts_text, '', prompt_speech_16k)
        # in cross lingual mode, we remove prompt in llm
        del model_input['prompt_text']
        del model_input['prompt_text_len']
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        """
        prepare input for instruct mode

        1. first, we prepare input for sft mode using tts_text and spk_id
        2. then, we remove spk_embedding in llm due to information leakage
        3. finally, we add instruct_text as prompt_text for llm
        """
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input['llm_embedding']
        # add instruct_text as prompt_text for llm
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input

