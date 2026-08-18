# Teste de contexto longo com o KV em q2_1 (2.25 bpw).
# Sem MTP: os ~2 GiB do contexto de draft nao cabem neste orcamento.
# YaRN e obrigatorio acima dos 262144 nativos.
$bin = "E:\DEV\turboquant\llama.cpp\build\bin\Release"
Write-Host "VRAM antes:" -ForegroundColor Cyan
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
& "$bin\llama-server.exe" `
  -m E:/models/Qwen3.8-27B-Q4_K_M.gguf `
  -a qwen3.8-27b `
  -c 450560 `
  -np 1 `
  -ctk q2_1 -ctv q2_1 `
  -fa on -ngl 99 -fit off `
  -b 512 -ub 512 `
  --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 `
  --log-file E:\DEV\turboquant\logs\server-450k.log `
  --host 127.0.0.1 --port 11434
