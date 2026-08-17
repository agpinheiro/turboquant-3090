# Teste de contexto cheio via API: 166194 tokens de prompt + pergunta de recall sobre o INICIO
# do texto (a parte mais distante na atencao). Mede prefill real e geracao com o KV em q4_0.
# Requer o server de run-200k-mtp.ps1 no ar em 127.0.0.1:8080.

param(
    [string]$File   = "E:\DEV\turboquant\data\warpeace-700k.txt",
    [string]$Out    = "E:\DEV\turboquant\logs\fullctx-result.json",
    [int]   $MaxTok = 512
)

$ErrorActionPreference = "Stop"

$text = Get-Content -Raw -Path $File
$q = @"
Pergunta sobre o texto acima. Responda em portugues, em no maximo 5 linhas, sem repetir o texto:
nas primeiras paginas, a narrativa comeca em um evento social. Em que cidade isso acontece,
que tipo de evento e, e qual o nome da anfitria?
"@

$payload = @{
    messages    = @(@{ role = "user"; content = ($text + "`n`n-----`n" + $q) })
    max_tokens  = $MaxTok
    temperature = 0
    stream      = $false
}

$json  = $payload | ConvertTo-Json -Depth 5 -Compress
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)

Write-Host ("payload: {0:N0} KB" -f ($bytes.Length/1KB))
$sw = [System.Diagnostics.Stopwatch]::StartNew()

$r = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/chat/completions" `
                       -Method Post -ContentType "application/json; charset=utf-8" `
                       -Body $bytes -TimeoutSec 5400

$sw.Stop()

$r | ConvertTo-Json -Depth 8 | Out-File -FilePath $Out -Encoding utf8

$t = $r.timings
"=== wall clock: {0:N1}s ===" -f $sw.Elapsed.TotalSeconds
"prompt   : {0:N0} tokens em {1:N1}s = {2:N1} t/s" -f $t.prompt_n, ($t.prompt_ms/1000), $t.prompt_per_second
"gen      : {0:N0} tokens em {1:N1}s = {2:N1} t/s" -f $t.predicted_n, ($t.predicted_ms/1000), $t.predicted_per_second
"draft    : {0}/{1}" -f $t.draft_n_accepted, $t.draft_n
"cache_n  : {0}" -f $t.cache_n
"`n=== resposta ===`n" + $r.choices[0].message.content
