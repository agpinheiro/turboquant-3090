# Teste manual interativo: Qwen3.8-27B Q4_K_M @ 200k de contexto, KV em q4_0, tudo na GPU.
# Estimativa (llama-fit-params): 15345 (modelo) + 3668 (KV) + 1057 (compute) = 20070 MiB de 24576.
# -fit off + -ngl 99 => se nao couber, morre com CUDA OOM em vez de escorrer para a RAM.

$bin   = "E:\DEV\turboquant\llama.cpp\build\bin\Release"
$model = "E:/models/Qwen3.8-27B-Q4_K_M.gguf"

Write-Host "VRAM antes do load:" -ForegroundColor Cyan
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

Write-Host "`nCarregando 16 GB do disco, aguarde... (/exit ou Ctrl+C para sair)`n" -ForegroundColor Cyan

& "$bin\llama-cli.exe" `
  -m $model `
  -c 200000 `
  -ctk q4_0 -ctv q4_0 `
  -fa on `
  -ngl 99 `
  -fit off `
  -b 512 -ub 512 `
  -cnv
