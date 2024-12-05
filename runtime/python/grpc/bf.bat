@echo off
:: 设置 UTF-8 编码
chcp 65001

:: 创建报告目录
if not exist "reports" mkdir reports

:: 获取当前时间戳
set "TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"

echo 开始性能测试...
echo 时间: %TIMESTAMP%

:: 运行测试并生成报告
ghz --insecure ^
--proto ./cosyvoice.proto ^
--call cosyvoice.CosyVoice.Inference ^
-D ./data.json ^
-n 1 ^
-c 1 ^
-O html ^
-o reports/report_%TIMESTAMP%.html ^
localhost:50000

:: 检查是否成功
if %ERRORLEVEL% EQU 0 (
    echo 测试完成！
    echo 报告已保存到: reports/report_%TIMESTAMP%.html
    :: 自动打开报告
    start "" "reports/report_%TIMESTAMP%.html"
) else (
    echo 测试执行失败！
)

pause