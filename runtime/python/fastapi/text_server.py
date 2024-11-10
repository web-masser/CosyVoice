from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import re
from typing import List, Dict
import time
import hanlp
import jieba
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class TextAnalyzer:
    def __init__(self):
        print("初始化分析器...")
        
        # 添加章节模式定义
        self.chapter_patterns = [
            r'^[^第]*第\d+章[^\n]*',           # 匹配 "☆、第1章"
            r'^[^第]*第[零一二三四五六七八九十百千万]+章[^\n]*',  # 匹配 "☆、第一章"
            r'^[^第]*第\d+回[^\n]*',           # 匹配 "☆、第1回"
            r'^[^第]*第[零一二三四五六七八九十百千万]+回[^\n]*',  # 匹配 "☆、第一回"
            r'^[^第]*第\d+节[^\n]*',           # 匹配 "☆、第1节"
            r'^[^第]*第[零一二三四五六七八九十百千万]+节[^\n]*',  # 匹配 "☆、第一节"
            r'^\d+[、.][^\n]*',          # 匹配 "1、" 或 "1."
            r'^Chapter\s*\d+[^\n]*',      # 匹配 "Chapter 1"
            r'^[零一二三四五六七八九十百千万]+、[^\n]*'  # 匹配 "一、"
        ]
        
        try:
            print("初始化 HanLP...")
            # 设置镜像源
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像站点
            self.nlp = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
            print("HanLP 初始化完成")
            
            # 测试NER
            test = self.nlp("张三和李四是好朋友。")
            print(f"测试结果: {test}")
        except Exception as e:
            print(f"HanLP 初始化失败: {str(e)}")
            raise

    def extract_characters(self, chapter: Dict) -> Dict:
        try:
            print(f"\n处理章节: {chapter['title']}")
            text = chapter['content']
            print(f"章节内容长度: {len(text)} 字符")
            characters = set()
            character_count = {}
            
            # 分段处理
            segments = []
            current_segment = ""
            for line in text.split('\n'):
                if len(current_segment) + len(line) < 100:
                    current_segment += line
                else:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = line
            if current_segment:
                segments.append(current_segment)
            
            print(f"分段数量: {len(segments)}")
            
            # 只处理 NER 结果
            for i, segment in enumerate(segments):
                try:
                    print(f"\n处理第 {i+1} 段:")
                    print(f"段落内容: {segment[:50]}...")
                    
                    # NER识别
                    ner_result = self.nlp(segment)
                    print(f"NER 原始结果: {ner_result}")
                    
                    # 只统计 NER 识别出的人名
                    for entity in ner_result:
                        word, tag = entity[0], entity[1]
                        # 检查是否为中文人名且长度大于1
                        if tag == 'NR' and re.search(r'[\u4e00-\u9fff]', word) and len(word) > 1:
                            characters.add(word)
                            character_count[word] = character_count.get(word, 0) + 1
                            print(f"找到人物: {word}, 当前出现次数: {character_count[word]}")
                                
                except Exception as e:
                    print(f"处理段落时出错: {str(e)}")
                    continue
            
            print("\n统计结果:")
            print(f"唯一人物数量: {len(characters)}")
            print(f"人物列表: {list(characters)}")
            print(f"出现次数统计: {character_count}")
            
            return {
                'characters': list(characters),
                'character_count': character_count,
                'total': len(characters)
            }
                
        except Exception as e:
            print(f"章节处理主错误: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def split_chapters(self, text: str) -> List[Dict]:
        """将文本分割成章节"""
        chapters = []
        lines = text.split('\n')
        current_chapter = {'title': '', 'content': ''}
        
        print("开始分析章节...")
        for line in lines:
            # 检查是否是章节标题
            is_chapter_title = False
            line = line.strip()
            if line:  # 只处理非空行
                for pattern in self.chapter_patterns:
                    if re.match(pattern, line):
                        # 如果当前章节不为空，保存它
                        if current_chapter['content']:
                            chapters.append(current_chapter)
                        # 创建新章节
                        current_chapter = {
                            'title': line,
                            'content': ''
                        }
                        is_chapter_title = True
                        break
        
            # 如果不是章节标题，添加到当前章节内容
            if not is_chapter_title and current_chapter['title']:
                current_chapter['content'] += line + '\n'
        
        # 添加最后一个章节
        if current_chapter['content']:
            chapters.append(current_chapter)
            
        print(f"找到章节数量: {len(chapters)}")
        if chapters:
            print("章节标题列表:")
            for i, chapter in enumerate(chapters):
                print(f"{i+1}. {chapter['title']}")
        else:
            print("未找到任何章节!")
        
        return chapters

    def extract_dialogs(self, chapter: Dict) -> Dict:
        try:
            print(f"\n处理章节对话: {chapter['title']}")
            text = chapter['content']
            
            # 定义更丰富的语气动词列表
            speech_verbs = [
                 # 话互动 - 基础
                '对', '向', '冲', '跟', '和', '与', '同', '给', '朝', '望', '看',
                
                # 对话互动 - 应答
                '回答', '答道', '应道', '应声', '回声', '搭腔', '接口', '接茬', '接话',
                '答话', '回话', '应话', '搭话', '接嘴', '答腔', '回应', '应答', '答应',
                
                # 对话互动 - 打断
                '打断', '截住', '拦住', '插嘴', '插话', '打岔', '截话', '拦话', '断话',
                '打住', '止住', '阻住', '遮住', '挡住', '堵住', '截断', '中断', '打住',
                
                # 对话互动 - 追问
                '追问', '追究', '追根', '追查', '盘问', '诘问', '质问', '详问', '细问',
                '反问', '反诘', '反击', '回击', '驳斥', '辩驳', '争辩', '辩解', '分辩',
                
                # 对话互动 - 商议
                '商议', '商量', '商讨', '商酌', '商榷', '商定', '商谈', '商说', '商论',
                '议论', '讨论', '协商', '磋商', '洽谈', '谈判', '交涉', '探讨', '研究',
                
                # 对话互动 - 争执
                '争执', '争论', '争辩', '争吵', '吵架', '斗嘴', '拌嘴', '顶嘴', '抬杠',
                '理论', '辩论', '争论', '争斗', '争吵', '吵闹', '争夺', '争抢', '争辩',
                
                # 对话互动 - 劝说
                '劝说', '劝告', '劝解', '劝导', '劝诫', '劝慰', '劝阻', '劝止', '规劝',
                '开导', '指导', '指点', '指教', '教导', '引导', '启发', '点拨', '提点',
                
                # 对话互动 - 安慰
                '安慰', '抚慰', '慰藉', '宽慰', '开解', '解劝', '劝慰', '劝解', '开导',
                '宽心', '放心', '安心', '定心', '宽怀', '释怀', '解忧', '解愁', '解闷',
                
                # 对话互动 - 恳求
                '恳求', '哀求', '央求', '乞求', '请求', '祈求', '恳请', '央请', '乞请',
                '求情', '讨情', '说情', '求饶', '求助', '求援', '求救', '求告', '求教',
                
                # 对话互动 - 命令
                '命令', '吩咐', '嘱咐', '交代', '指示', '指令', '下令', '示意', '吩咐',
                '叮嘱', '叮咛', '嘱托', '嘱告', '告诫', '训诫', '训斥', '呵斥', '斥责',
                
                # 对话互动 - 商榷
                '商榷', '商酌', '商议', '商量', '商讨', '商定', '商谈', '商说', '论',
                '讨论', '议论', '协商', '磋商', '洽谈', '谈判', '交涉', '探讨', '研究',
                
                # 对话互动 - 组合表达
                '对着说', '向着问', '冲着喊', '跟着答', '和着笑', '给着讲', '朝着道',
                '对面说', '当面问', '面对面', '背对背', '你一言', '我一语', '一唱一和',
                '一问一答', '一来一往', '一说一笑', '一边说', '一面说', '一直说',
                # 基础表达
                '说', '道', '问', '答', '讲', '言', '谈', '述', '话', '聊', '告诉', '表示',
                
                # 声音描述
                '喊', '叫', '吼', '嚷', '喝', '呼', '嘶', '吟', '嚎', '喃', '嘀', '嘟',
                '哼', '啧', '咳', '叹', '笑', '哭', '骂', '喘', '哈', '呵', '嘻', '嘿',
                '咦', '哦', '哎', '呀', '啊', '唉', '哟', '呢', '吧', '呸', '咯', '嗯',
                '哇', '呜', '嗨', '嘘', '嗬', '咦', '咧', '嗝', '嗤', '啐', '噗', '咕',
                
                # 动作描述
                '指', '比', '摆', '晃', '摇', '点', '瞥', '瞪', '瞧', '看', '望', '盯',
                '扫', '瞄', '觑', '瞅', '睨', '视', '望', '顾', '睇', '眄', '瞠', '眺',
                '凝', '眈', '眸', '盼', '眨', '眯', '睁', '闭', '眼', '目', '视', '睹',
                
                # 身体动作
                '站', '坐', '蹲', '跪', '躺', '趴', '俯', '仰', '侧', '转', '回', '走',
                '跑', '跳', '蹦', '踱', '溜', '窜', '奔', '冲', '闪', '掠', '飘', '飞',
                '扑', '扯', '拽', '拉', '推', '搡', '抱', '搂', '抓', '握', '拍', '摸',
                '碰', '触', '戳', '捅', '捏', '掐', '掰', '揉', '搓', '擦', '抖', '晃',
                
                # 面部表情
                '笑', '哭', '愁', '皱', '蹙', '横', '瞪', '眯', '眨', '睁', '闭', '咧',
                '撇', '抿', '咬', '吐', '舔', '啃', '嚼', '吮', '吸', '呼', '喷', '吹',
                
                # 情绪状态
                '喜', '怒', '哀', '乐', '愁', '烦', '忧', '虑', '怯', '惧', '慌', '急',
                '躁', '静', '安', '闲', '慵', '懒', '倦', '困', '醒', '睡', '醉', '醒',
                
                # 心理活动
                '想', '思', '忆', '记', '觉', '悟', '懂', '晓', '知', '解', '猜', '疑',
                '虑', '忧', '盼', '望', '期', '待', '信', '疑', '惑', '惘', '迷', '悟',
                
                # 语气状态
                '急', '慢', '缓', '徐', '快', '疾', '迅', '速', '匆', '忙', '赶', '慌',
                '从容', '悠然', '徐徐', '缓缓', '慢慢', '渐渐', '款款', '徐缓', '迟迟',
                
                # 复合表达
                '开口', '出声', '发声', '开腔', '接话', '搭话', '插话', '打断', '截住',
                '回应', '应答', '答复', '回复', '反问', '追问', '质问', '诘问', '盘问',
                
                # 特殊表达
                '传音', '传声', '传话', '传言', '传述', '传达', '示意', '暗示', '明示',
                '透露', '流露', '表露', '显露', '露出', '显出', '现出', '浮现', '呈现',
                
                # 常见组合（动词+介词/助词）
                '说着', '说了', '说过', '说到', '说起', '说出', '说给', '说与', '说向',
                '问着', '问了', '问过', '问到', '问起', '问出', '问给', '问与', '问向',
                '答着', '答了', '答过', '答到', '答起', '答出', '答给', '答与', '答向',
                
                # 带标点的表达
                '：', '，', '！', '？', '；', '。',
                
                # 新增：姿态描述
                '挺', '弓', '曲', '弯', '直', '歪', '斜', '倾', '倚', '靠', '依', '偎',
                '蜷', '缩', '伸', '展', '舒', '张', '绷', '松', '紧', '僵', '软', '硬',
                
                # 新增：气息描述
                '吸', '呼', '喘', '吐', '憋', '屏', '噎', '咽', '吞', '呛', '咳', '嗽',
                
                # 新增：目光描述
                '瞟', '瞥', '瞄', '瞧', '瞪', '眯', '眨', '睁', '闭', '盯', '望', '看',
                '视', '观', '瞻', '眺', '望', '顾', '睐', '睨', '瞠', '眄', '盼', '眸',
                
                # 新增：手部动作
                '握', '抓', '捏', '掐', '拧', '扭', '拽', '扯', '拉', '推', '搡', '抱',
                '搂', '抄', '摸', '碰', '触', '戳', '捅', '掰', '揉', '搓', '擦', '抖',
                
                # 新增：情感表达
                '欢', '喜', '乐', '悦', '忧', '愁', '悲', '恨', '怒', '惧', '怯', '慌',
                '羞', '臊', '窘', '急', '躁', '静', '安', '闲', '慵', '懒', '倦', '困',
                
                # 常见组合
                '说道', '说着', '说完', '说了', '说过', '说出', '说到',
                '问道', '问着', '问完', '问了', '问过', '问起',
                '答道', '答应', '答复', '答话', '回答', '回应', '回道',
                '开口', '出声', '发声', '开腔', '接话', '搭话', '插话',
                
                # 音量相关
                '喊', '叫', '吼', '嚷', '喝', '呼', '嘶', '吟',
                '低语', '轻语', '细语', '轻声', '低声', '小声', '高声', '大声',
                '呐喊', '怒吼', '咆哮', '呢喃', '喃喃', '嘀咕', '嘟囔', '念叨',
                
                # 情绪相关
                '笑', '笑着', '笑了', '笑起', '笑道', '大笑', '微笑', '冷笑',
                '怒', '怒气', '怒火', '怒意', '怒形', '怒容', '怒色', '怒目',
                '哭', '哭着', '哭了', '哭起', '哭泣', '啜泣', '抽泣', '呜咽',
                '叹', '叹息', '叹气', '感叹', '惊叹', '赞叹', '惋叹', '长叹',
                
                # 动作描述
                '点头', '摇头', '挥手', '摆手', '伸手', '抬手', '举手', '挥动',
                '转身', '回头', '侧身', '俯身', '仰头', '低头', '抬头', '侧头',
                '皱眉', '蹙眉', '挑眉', '眨眼', '瞪眼', '眯眼', '闭眼', '睁眼',
                '张口', '咧嘴', '撇嘴', '抿嘴', '咬牙', '磨牙', '咽口', '吞口',
                
                # 表情描述
                '露出', '浮现', '流露', '展��', '显出', '呈现', '表现', '显露',
                '面带', '面露', '面现', '面有', '面上', '脸上', '神色', '神情',
                
                # 语气词（包含标点）
                '：', '，', '！', '？', '；', '。',
                
                # 复合表达
                '开口说', '出声道', '大声喊', '小声说', '轻声问', '低声答',
                '笑着说', '哭着说', '怒气道', '叹息道', '点头说', '摇头道',
                '转身说', '回头道', '皱眉问', '眯眼道', '张口答', '露出笑容',
                
                # 特殊表达
                '表示', '表达', '表态', '示意', '暗示', '明示', '透露', '流露',
                '传达', '传递', '传音', '传声', '传话', '传言', '传述', '传达',
                
                # 心理活动
                '心想', '心道', '心中', '心里', '想着', '思索', '思考', '琢磨',
                '寻思', '考虑', '回忆', '记起', '想起', '忆起', '记得', '觉得',
                
                # 语气状态
                '急忙', '连忙', '赶忙', '慌忙', '匆忙', '忙着', '立刻', '马上',
                '缓缓', '慢慢', '徐徐', '渐渐', '慢声', '轻缓', '从容', '悠悠',
                
                # 情感状态
                '欣喜', '焦急', '愤怒', '惊讶', '疑惑', '好奇', '担忧', '紧张',
                '笑道', '怒道', '骂道', '哭道', '叹道', '叹息道', '感叹道', '惊道', '咋舌道',
                '惊呼道', '赞道', '夸道', '讽道', '道', '嘲道', '怪道', '悲道', '喜道',
                '乐道', '苦笑道', '冷笑道', '嗤笑道', '大笑道', '狂笑道', '微笑道', '哈哈道',
                '嘿嘿道', '呵呵道', '嘻嘻道', '哼道', '哼哼道', '啧道', '啧啧道',
                
                # 语速相关
                '急道', '慢道', '缓道', '徐徐道', '徐道', '慢慢道', '急急道', '快速道',
                '迅速道', '立刻道', '马上道', '连忙道', '赶紧道', '赶快道', '忙道',
                
                # 态度相关
                '正色道', '严肃道', '认真道', '诚恳道', '和气道', '温和道', '客气道', 
                '恭敬道', '谦逊道', '谦和道', '傲慢道', '轻蔑道', '不屑道', '严厉道',
                '温柔道', '和蔼道', '亲切道', '慈祥道', '和善道', '和气道', '冷冷道',
                '淡淡道', '冰冷道', '热情道', '激动道', '兴奋道', '平静道', '淡定道',
                
                # 思考相关
                '想道', '思道', '考虑道', '回忆道', '猜测道', '分析道', '判断道',
                
                # 动作相关
                '点头道', '摇头道', '挥手道', '摆手道', '伸手道', '抬手道', '举手道',
                '转身道', '回道', '侧身道', '俯身道', '仰头道', '低头道', '抬头道',
                '皱眉道', '蹙眉道', '挑眉道', '眨眼道', '瞪眼道', '眯眼道', '闭眼道',
                '张口道', '咧嘴道', '撇嘴道', '抿嘴道', '咬牙道', '磨牙道',
                
                # 复合情绪动作
                '欣喜道', '焦急道', '愤怒道', '惊讶道', '疑惑道', '好奇道', '担忧道',
                '紧张道', '害怕道', '恐惧道', '羞涩道', '尴尬道', '无奈道', '失望道',
                '沮丧道', '开心道', '兴奋道', '激动道', '平淡道', '冷静道', '镇定道',
                
                # 语气词
                '继续道', '补充道', '插话道', '打断道', '接话道', '附和道', '应道',
                '反驳道', '争辩道', '解释道', '劝道', '提醒道', '建议道', '命令道',
                '吩咐道', '催促道', '恳求道', '央求道', '哀求道', '感激道', '称赞道',
                
                # 特殊语气
                '嘟囔道', '嘀咕道', '感慨道', '惊叹道', '疑道', '诧异道', '追问道',
                '反问道', '质问道', '调侃道', '打趣道', '玩笑道', '讨好道', '撒娇道',
                '抱怨道', '责备道', '训斥道', '呵斥道', '斥责道', '咒骂道', '诅咒道',
                
                # 常见组合
                '开口道', '出声道', '大喊道', '小声道', '轻叹道', '重复道', '强调道',
                '附加道', '总结道', '分析道', '评价道', '评论道', '感叹道', '惊呼道',
                
                # 带标点的变体
                '说：', '道：', '问：', '答：', '喊：', '叫：', '笑：', '怒：',
                '说，', '道，', '问，', '答，', '喊，', '叫，', '笑，', '怒，'
            ]
            
            # 先获取人物列表
            character_result = self.extract_characters(chapter)
            characters = character_result['characters']
            print(f"已知人物列表: {characters}")
            
            # 先保护引号内的句号
            protected_text = re.sub(r'\u201c([^\u201d]*)\u201d', 
                lambda m: '\u201c' + m.group(1).replace('。', '<DOT>') + '\u201d', 
                text)
            
            # 按句号分段
            paragraphs = protected_text.split('。')
            paragraphs = [p.replace('<DOT>', '。').strip() for p in paragraphs if p.strip()]
            
            print(f"段落数量: {len(paragraphs)}")
            
            def find_nearest_character(text: str, characters: list, start_pos: int, direction: str = 'before') -> tuple:
                """
                查找最近的人物名称
                direction: 'before' 向前查找, 'after' 向后查找
                返回: (人物名称, 是否有语气词)
                """
                if direction == 'before':
                    # 获取对话前的文本
                    search_text = text[:start_pos]
                    # 找到最后出现的人物名称
                    last_char = None
                    last_pos = -1
                    has_speech_verb = False
                    
                    for char in characters:
                        pos = search_text.rfind(char)
                        if pos > last_pos:
                            # 检查人名和对话之间是否有语气词
                            text_between = search_text[pos+len(char):start_pos]
                            has_verb = any(verb in text_between for verb in speech_verbs)
                            if has_verb or last_char is None:  # 如果有语气词或还没找到人物
                                last_char = char
                                last_pos = pos
                                has_speech_verb = has_verb
                    
                    return last_char, has_speech_verb
                else:
                    # 获取对话后的文本
                    search_text = text[start_pos:]
                    # 找到最先出现的人物名称
                    first_char = None
                    first_pos = len(search_text)
                    has_speech_verb = False
                    
                    for char in characters:
                        pos = search_text.find(char)
                        if pos != -1 and pos < first_pos:
                            # 检查对话和人名之间是否有语气词
                            text_between = search_text[:pos]
                            has_verb = any(verb in text_between for verb in speech_verbs)
                            if has_verb or first_char is None:  # 如果有语气词或还没找到人物
                                first_char = char
                                first_pos = pos
                                has_speech_verb = has_verb
                    
                    return first_char, has_speech_verb
            
            dialogs = []
            for para in paragraphs:
                # 在段落中查找对话
                dialog_matches = re.finditer(r'\u201c(.*?)\u201d', para)
                
                for match in dialog_matches:
                    dialog_content = match.group(1)
                    dialog_start = match.start()
                    dialog_end = match.end()
                    
                    # 先查找前文中带语气词的人物
                    speaker_before, has_verb_before = find_nearest_character(para, characters, dialog_start, 'before')
                    # 再查找后文中带语气词的人物
                    speaker_after, has_verb_after = find_nearest_character(para, characters, dialog_end, 'after')
                    
                    # 确定说话人
                    speaker = None
                    if has_verb_before:  # 优先使用带语气词的前文人物
                        speaker = speaker_before
                    elif has_verb_after:  # 其次使用带语气词的后文人物
                        speaker = speaker_after
                    else:  # 如果都没有语气词，使用最近的人物
                        # 计算距离，选择最近的
                        dist_before = dialog_start - para.rfind(speaker_before) if speaker_before else float('inf')
                        dist_after = para.find(speaker_after, dialog_end) if speaker_after else float('inf')
                        speaker = speaker_before if dist_before < dist_after else speaker_after
                    
                    dialog_info = {
                        "content": dialog_content,
                        "speaker": speaker,
                        "context": para,
                        "has_speech_verb": has_verb_before or has_verb_after
                    }
                    dialogs.append(dialog_info)
                    print(f"\n对话: {dialog_content}")
                    print(f"说话人: {speaker}")
                    print(f"是否有语气词: {dialog_info['has_speech_verb']}")
                    print(f"段落: {para[:50]}...")
            
            return {
                'dialogs': dialogs,
                'total': len(dialogs)
            }
                
        except Exception as e:
            print(f"处理对话时出错: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# 初始化分析器
analyzer = TextAnalyzer()

@app.post("/extract_characters")
async def extract_characters(file: UploadFile = File(...)):
    try:
        print(f"\n收到文件: {file.filename}")
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="只支持txt文件")

        content = await file.read()
        print(f"文件大小: {len(content)} 字节")

        # 尝试多种编码解码
        text = None
        for encoding in ['utf-8', 'gbk', 'gb2312']:
            try:
                text = content.decode(encoding)
                print(f"使用 {encoding} 解码成功")
                break
            except:
                continue

        if text is None:
            raise HTTPException(status_code=400, detail="无法识别文件编码")

        # 分割章节
        chapters = analyzer.split_chapters(text)
        
        if not chapters:
            return {
                "filename": file.filename,
                "error": "未找到任何章节"
            }
            
        # 只处理第一个章节
        first_chapter = chapters[0]
        result = analyzer.extract_characters(first_chapter)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chapter_title": first_chapter["title"],
            "characters": result['characters'],
            "character_count": result['character_count'],
            "total": result['total']
        }

    except Exception as e:
        print(f"处理错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/extract_dialogs")
async def extract_dialogs(file: UploadFile = File(...)):
    try:
        print(f"\n收到文件: {file.filename}")
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="只支持txt文件")

        content = await file.read()
        print(f"文件大小: {len(content)} 字节")

        # 尝试多种编码解码
        text = None
        for encoding in ['utf-8', 'gbk', 'gb2312']:
            try:
                text = content.decode(encoding)
                print(f"使用 {encoding} 解码成功")
                break
            except:
                continue

        if text is None:
            raise HTTPException(status_code=400, detail="无法识别文件编码")

        # 分割章节
        chapters = analyzer.split_chapters(text)
        
        if not chapters:
            return {
                "status": "error",
                "message": "未找到任何章节",
                "filename": file.filename
            }
            
        # 只处理第一个章节
        first_chapter = chapters[0]
        result = analyzer.extract_dialogs(first_chapter)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chapter_title": first_chapter["title"],
            "dialogs": result['dialogs'],
            "total": result['total']
        }

    except Exception as e:
        print(f"处理错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/extract_dialogs")
async def websocket_extract_dialogs(websocket: WebSocket):
    print("尝试建立 WebSocket 连接...")
    try:
        await websocket.accept()
        print("WebSocket 连接已接受")
        
        print("等待接收数据...")
        data = await websocket.receive_text()
        print(f"收到数据: {data}")
        
        file_data = json.loads(data)
        text = file_data['content']
        chapter_index = file_data.get('chapter_index', 0)
        
        # 分割章节
        chapters = analyzer.split_chapters(text)
        selected_chapter = chapters[chapter_index]
        chapter_text = selected_chapter['content']
        
        print(f"章节内容长度: {len(chapter_text)}")
        
        # 获取人物列表和对话结果
        dialog_result = analyzer.extract_dialogs(selected_chapter)
        
        # 发送每个对话
        for dialog in dialog_result['dialogs']:
            dialog_info = {
                "type": "dialog",
                "content": dialog
            }
            print(f"发送对话: {dialog_info}")
            await websocket.send_text(json.dumps(dialog_info, ensure_ascii=False))
        
        print(f"总共找到 {dialog_result['total']} 个对话")
        # 发送完成消息
        await websocket.send_text(json.dumps({
            "type": "complete",
            "total": dialog_result['total']
        }, ensure_ascii=False))

    except Exception as e:
        print(f"WebSocket 错误: {str(e)}")
        print(f"错误类型: {type(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }, ensure_ascii=False))
    finally:
        print("WebSocket 连接关闭")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("text_server:app", host="0.0.0.0", port=6710, reload=True)