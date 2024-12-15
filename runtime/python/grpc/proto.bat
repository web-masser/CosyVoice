@echo off
setlocal

set OUT_DIR=.\utils

if not exist %OUT_DIR% mkdir %OUT_DIR%

protoc --plugin=protoc-gen-ts=.\node_modules\.bin\protoc-gen-ts.cmd ^
  --js_out=import_style=commonjs,binary:%OUT_DIR% ^
  --ts_out=service=grpc-web:%OUT_DIR% ^
  --proto_path=. ^
  cosyvoice.proto

echo Generation completed!