param([int]$NMax = 2)   # --spec-draft-n-max; so vale no startup, o campo por requisicao e ignorado

# Teste manual: mesmo modelo/contexto do run-200k.ps1, mas com MTP speculative decoding.
# MTP so funciona no llama-server (o llama-cli nao cria ctx_dft).
# Nao carrega um segundo modelo: cria um contexto LLAMA_CONTEXT_TYPE_MTP sobre os mesmos pesos.
# -ctkd/-ctvd q4_0 porque o KV do contexto de draft nasce em f16 por padrao.
# -np 1 porque o server usa 4 slots por padrao, e cada slot custa buffer.

$bin   = "E:\DEV\turboquant\llama.cpp\build\bin\Release"
$model = "E:/models/Qwen3.8-27B-Q4_K_M.gguf"
$log   = "E:\DEV\turboquant\logs\server-mtp.log"

Write-Host "VRAM antes do load:" -ForegroundColor Cyan
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
Write-Host "`nUI em http://127.0.0.1:8080 quando carregar. Ctrl+C so quando quiser encerrar.`n" -ForegroundColor Cyan

& "$bin\llama-server.exe" `
  -m $model `
  -a "qwen3.8-27b" `
  -c 200000 `
  -np 1 `
  -ctk q4_0 -ctv q4_0 `
  -ctkd q4_0 -ctvd q4_0 `
  -fa on `
  -ngl 99 `
  -fit off `
  -b 512 -ub 512 `
  --spec-type draft-mtp `
  --spec-draft-n-max $NMax `
  --log-file $log `
  --host 127.0.0.1 --port 8080
