@echo off
REM 以管理员权限运行
powershell Start-Process cmd -Verb RunAs -ArgumentList '/k "cd /d D:\project\CosyVoice\runtime\python\fastapi && conda activate cosyvoice && python server.py"'
exit