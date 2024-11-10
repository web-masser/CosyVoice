import streamlit as st
import requests
import json
from pathlib import Path
import logging
import pandas as pd
import sys
import websockets
import asyncio

class TextAnalysisClient:
    def __init__(self, base_url: str = "ws://127.0.0.1:6710"):
        self.base_url = base_url.rstrip('/')
        self.ws_url = f"{self.base_url}/ws/extract_dialogs"
        self.logger = self._setup_logger()
        self.timeout = 30
        self.chapters = []  # 存储章节信息

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("TextAnalysisClient")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def health_check(self) -> bool:
        """检查服务是否正常运行"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException as e:
            self.logger.error(f"健康检查失败: {str(e)}")
            return False

    def extract_characters(self, file) -> dict:
        """上传文件并提取人物角色"""
        try:
            files = {'file': file}
            response = requests.post(
                f"{self.base_url}/extract_characters",
                files=files,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API请求失败: HTTP {response.status_code}")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"API请求异常: {str(e)}")
            st.error(f"API请求异常: {str(e)}")
            return None

    def _detect_encoding(self, content: bytes) -> str:
        """检测文件编码"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'utf-16le', 'utf-16be']
        for encoding in encodings:
            try:
                content.decode(encoding)
                print(f"检测到编码: {encoding}")
                return encoding
            except UnicodeDecodeError:
                continue
        return 'utf-8'  # 默认返回utf-8

    async def get_chapters(self, file_content: bytes):
        """获取章节列表"""
        try:
            encoding = self._detect_encoding(file_content)
            text_content = file_content.decode(encoding)
            
            async with websockets.connect(f"{self.base_url}/ws/get_chapters") as websocket:
                await websocket.send(json.dumps({
                    "content": text_content,
                    "encoding": encoding
                }, ensure_ascii=False))
                
                response = await websocket.recv()
                data = json.loads(response)
                if data["status"] == "success":
                    self.chapters = data["chapters"]
                    return self.chapters
                else:
                    st.error(f"获取章节失败: {data['message']}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"获取章节错误: {str(e)}")
            st.error(f"获取章节错误: {str(e)}")
            return []

    async def extract_dialogs_stream(self, file_content: bytes, chapter_index: int):
        """通过WebSocket流式提取对话"""
        try:
            encoding = self._detect_encoding(file_content)
            text_content = file_content.decode(encoding)
            
            async with websockets.connect(self.ws_url) as websocket:
                await websocket.send(json.dumps({
                    "content": text_content,
                    "encoding": encoding,
                    "chapter_index": chapter_index
                }, ensure_ascii=False))
                
                # 创建进度条和状态文本
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 创建一个空的数据框
                df = pd.DataFrame(columns=["序号", "说话人", "对话内容", "上下文"])
                table_placeholder = st.empty()
                
                dialog_count = 0
                
                while True:
                    try:
                        response = await websocket.recv()
                        data = json.loads(response)
                        
                        if data["type"] == "chapter_info":
                            status_text.text(f"正在处理章节: {data['title']}")
                        
                        elif data["type"] == "dialog":
                            dialog_count += 1
                            dialog_info = data["content"]
                            
                            # 添加新的一行到数据框
                            new_row = pd.DataFrame([{
                                "序号": dialog_count,
                                "说话人": dialog_info["speaker"] or "未知",
                                "对话内容": f'"{dialog_info["content"]}"',
                                "上下文": dialog_info["context"][:50] + "..." if len(dialog_info["context"]) > 50 else dialog_info["context"]
                            }])
                            
                            # 将新行添加到数据框
                            df = pd.concat([df, new_row], ignore_index=True)
                            
                            # 更新表格显示
                            table_placeholder.dataframe(
                                df,
                                height=400,
                                use_container_width=True
                            )
                            
                            # 更新进度（假设总数为100）
                            progress_bar.progress(min(dialog_count, 100) / 100)
                        
                        elif data["type"] == "progress":
                            progress = min(data["processed"] / 100, 1.0)
                            progress_bar.progress(progress)
                        
                        elif data["type"] == "complete":
                            progress_bar.progress(100)
                            status_text.text(f"处理完成，共找到 {data['total']} 段对话")
                            
                            # 添加导出按钮
                            if not df.empty:
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="导出为CSV",
                                    data=csv,
                                    file_name="dialogs.csv",
                                    mime="text/csv"
                                )
                            break
                        
                        elif data["type"] == "error":
                            st.error(f"处理错误: {data['message']}")
                            break
                    
                    except websockets.exceptions.ConnectionClosed:
                        st.error("WebSocket连接已关闭")
                        break
                    
                    # 添加一个小延迟，让界面有时间更新
                    await asyncio.sleep(0.1)
                
                return df

        except Exception as e:
            self.logger.error(f"WebSocket连接错误: {str(e)}")
            st.error(f"连接错误: {str(e)}")
            return None

def main():
    st.title("文本分析工具")
    
    uploaded_file = st.file_uploader("选择文本文件", type=['txt'])
    
    if uploaded_file is not None:
        file_content = uploaded_file.read()
        
        # 检测文件编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'utf-16le']
        text_content = None
        for encoding in encodings:
            try:
                text_content = file_content.decode(encoding)
                print(f"使用 {encoding} 解码成功")
                break
            except UnicodeDecodeError:
                continue
        
        if text_content is None:
            st.error("无法识别文件编码")
            return
        
        if st.button("提取对话"):
            # 创建一个空的数据框显示区域
            table_placeholder = st.empty()
            dialogs = []
            
            # 创建WebSocket连接并实时显示结果
            async def process_dialogs():
                print("开始连接WebSocket...")
                async with websockets.connect("ws://localhost:6710/ws/extract_dialogs") as websocket:
                    print("WebSocket连接成功")
                    
                    # 发送文本内容
                    await websocket.send(json.dumps({
                        "content": text_content,
                        "chapter_index": 0
                    }))
                    print("已发送文本内容")
                    
                    while True:
                        try:
                            print("等待接收数据...")
                            response = await websocket.recv()
                            print(f"收到数据: {response}")
                            
                            data = json.loads(response)
                            print(f"数据类型: {data['type']}")
                            
                            if data["type"] == "dialog":
                                dialog = data["content"]
                                print(f"找到对话: {dialog['content']}")
                                
                                dialogs.append({
                                    "序号": len(dialogs) + 1,
                                    "说话人": dialog["speaker"] or "未知",
                                    "对话内容": dialog["content"],
                                    "上下文": dialog["context"]
                                })
                                
                                # 更新表格显示
                                df = pd.DataFrame(dialogs)
                                table_placeholder.dataframe(df)
                                print(f"当前对话数: {len(dialogs)}")
                                
                            elif data["type"] == "complete":
                                print("处理完成")
                                st.success(f"处理完成，共找到 {len(dialogs)} 个对话")
                                break
                                
                            elif data["type"] == "error":
                                print(f"处理错误: {data['message']}")
                                st.error(f"处理错误: {data['message']}")
                                break
                                
                        except websockets.exceptions.ConnectionClosed:
                            print("WebSocket连接已关闭")
                            st.error("WebSocket连接已关闭")
                            break
                        except Exception as e:
                            print(f"处理错误: {str(e)}")
                            st.error(f"处理错误: {str(e)}")
                            break
            
            # 运行WebSocket连接
            asyncio.run(process_dialogs())

if __name__ == "__main__":
    main()