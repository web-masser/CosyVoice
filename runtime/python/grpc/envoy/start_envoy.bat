@echo off
REM =============================================
REM 启动 Envoy Docker 容器的批处理脚本
REM =============================================

setlocal enabledelayedexpansion
set "ERROR_MSG="

REM 定义 Envoy 配置文件和证书的路径
set "ENVOY_CONFIG=%~dp0envoy.yaml"
set "ENVOY_CERTS=%~dp0certs"
set "ENVOY_PROTOS=%~dp0protos"

REM 检查 Envoy 容器是否已存在
docker ps -a --filter "name=envoy" --format "{{.Names}}" | findstr /I "^envoy$" >nul
IF NOT ERRORLEVEL 1 (
    echo 已存在名为 envoy 的容器，正在停止并移除...
    docker stop envoy
    docker rm envoy
)

REM 启动 Envoy 容器
echo 正在启动 Envoy 容器...
docker run --name envoy ^
  -d ^
  -p 6712:6712 ^
  --add-host=host.docker.internal:host-gateway ^
  -v "%ENVOY_CONFIG%:/etc/envoy/envoy.yaml" ^
  -v "%ENVOY_CERTS%:/etc/envoy/certs" ^
  -v "%ENVOY_PROTOS%:/etc/envoy/protos" ^
  envoyproxy/envoy:v1.32.1 || set "ERROR_MSG=启动容器失败"

REM 检查 Envoy 容器是否成功启动
docker ps --filter "name=envoy" --format "{{.Names}} 状态: {{.Status}}" || set "ERROR_MSG=检查容器状态失败"
IF %ERRORLEVEL%==0 (
    echo Envoy 容器已成功启动并正在运行。
    echo 正在检查容器日志...
    timeout /t 2 /nobreak >nul
    docker logs envoy
) ELSE (
    echo Envoy 容器启动失败，请检查日志。
)

REM 输出错误信息并暂停
if not "!ERROR_MSG!"=="" (
    echo !ERROR_MSG!
    pause
)

echo.
echo 按任意键查看实时日志，或直接关闭窗口...
pause >nul
docker logs -f envoy