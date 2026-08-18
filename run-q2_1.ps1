# Servidor para uso manual com o KV em q2_1 (2.25 bpw).
# 262144 e o teto do servidor: ele limita o slot ao contexto de treino do modelo
# (server-context.cpp:1202) e ignora o YaRN. Acima disso, use llama-completion.
# Como o q2_1 usa metade do KV do q4_0, sobra VRAM para o MTP - que o q4_0
# nao permitiria neste contexto.
param([int]$Port = 11434, [int]$Ctx = 262144)
$bin = "E:\DEV\turboquant\llama.cpp\build\bin\Release"
Write-Host "VRAM antes:" -ForegroundColor Cyan
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
Write-Host "`nUI em http://127.0.0.1:$Port  |  rede: http://192.168.0.3:$Port`n" -ForegroundColor Cyan
& "$bin\llama-server.exe" `
  -m E:/models/Qwen3.8-27B-Q4_K_M.gguf `
  -a qwen3.8-27b `
  --chat-template-file "E:\DEV\turboquant\qwen35-tolerant.jinja" `
  -c $Ctx -np 1 `
  -ctk q2_1 -ctv q2_1 `
  -ctkd q2_1 -ctvd q2_1 `
  -fa on -ngl 99 -fit off `
  -b 512 -ub 512 `
  --spec-type draft-mtp --spec-draft-n-max 2 `
  --log-file E:\DEV\turboquant\logs\server-q2_1.log `
  --host 0.0.0.0 --port $Port
