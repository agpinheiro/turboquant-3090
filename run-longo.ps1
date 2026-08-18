# Contexto longo com KV q2_1. Substitui o antigo run-450k-q2_1.ps1.
#
# Duas configuracoes uteis, medidas:
#   -Ctx 327680 -Mtp   -> ~1.3 GB livres, MTP funciona, e a config segura
#   -Ctx 450560        -> contexto maximo, sem MTP, ~9.5 t/s com o contexto cheio
#
# NAO use 450560 com -Mtp: carrega deixando 530 MiB livres, mas durante a geracao
# cai para ~320 e o WDDM comeca a paginar. A requisicao trava. Medido.
#
# Acima de 262144 o YaRN e obrigatorio, e depende do patch em server-context.cpp
# (branch kv-q2_0) que impede o servidor de capar o slot no contexto de treino.
param(
    [int]$Ctx  = 327680,
    [int]$Port = 11434,
    [switch]$Mtp
)
$bin = "E:\DEV\turboquant\llama.cpp\build\bin\Release"
$spec = if ($Mtp) { @("--spec-type","draft-mtp","--spec-draft-n-max","2","-ctkd","q2_1","-ctvd","q2_1") } else { @() }
$yarn = if ($Ctx -gt 262144) { @("--rope-scaling","yarn","--rope-scale","4","--yarn-orig-ctx","262144") } else { @() }

Write-Host "VRAM antes:" -ForegroundColor Cyan
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
Write-Host ("`nctx {0} | KV q2_1 | MTP {1} | YaRN {2}" -f $Ctx, $(if($Mtp){"on"}else{"off"}), $(if($yarn){"on"}else{"off"})) -ForegroundColor Cyan
Write-Host "UI em http://127.0.0.1:$Port  |  rede: http://192.168.0.3:$Port`n" -ForegroundColor Cyan

& "$bin\llama-server.exe" `
  -m E:/models/Qwen3.8-27B-Q4_K_M.gguf `
  -a qwen3.8-27b `
  --chat-template-file "E:\DEV\turboquant\qwen35-tolerant.jinja" `
  -c $Ctx -np 1 `
  -ctk q2_1 -ctv q2_1 `
  -fa on -ngl 99 -fit off `
  -b 512 -ub 512 `
  @spec @yarn `
  --log-file E:\DEV\turboquant\logs\server-longo.log `
  --host 0.0.0.0 --port $Port
