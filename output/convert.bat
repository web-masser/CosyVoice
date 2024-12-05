@echo off
setlocal

:: 设置输入和输出目录
set input_dir=.
set output_dir=converted

:: 创建输出目录（如果不存在）
if not exist "%output_dir%" mkdir "%output_dir%"

:: 转换 WAV 文件为 MOV
for %%f in ("%input_dir%\*.wav") do (
    ffmpeg -i "%%f" -codec:a aac -b:a 192k -f mov "%output_dir%\%%~nf.mov"
)

:: 转换 MP4 文件为 MOV
for %%f in ("%input_dir%\*.mp4") do (
    ffmpeg -i "%%f" -codec:v libx264 -profile:v high -level:v 4.0 -preset slow -crf 22 -codec:a aac -b:a 192k -movflags +faststart "%output_dir%\%%~nf.mov"
)

:: 转换 MOV 文件为兼容格式
for %%f in ("%input_dir%\*.mov") do (
    ffmpeg -i "%%f" -codec:v libx264 -profile:v high -level:v 4.0 -preset slow -crf 22 -codec:a aac -b:a 192k -movflags +faststart "%output_dir%\%%~nf.mov"
)

endlocal