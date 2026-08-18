# Mede um .gguf para decidir troca de quantizacao de pesos.
#
# Responde tres coisas que so se sabem com o arquivo na mao:
#   1. quanto do modelo vai para CUDA0 e quanto fica no host (nao da para inferir do tamanho)
#   2. qual o teto real de contexto com o KV disponivel
#   3. quanto de qualidade se perde contra o modelo de referencia
#
# Uso: .\medir-modelo.ps1 -Model E:/models/Qwen3.8-27B-Q3_K_M.gguf

param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Ref     = "E:/models/Qwen3.8-27B-Q4_K_M.gguf",
    [string]$Corpus  = "E:\DEV\turboquant\data\warpeace.txt",
    [int]   $PplCtx  = 32768,
    [int]   $Chunks  = 8,
    [switch]$SkipPpl
)

$ErrorActionPreference = "Continue"
$bin = "E:\DEV\turboquant\llama.cpp\build\bin\Release"

# VRAM disponivel de verdade: total - desktop - margem de seguranca de 800 MiB
$desktop = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits))
$budget  = 24576 - $desktop - 800
Write-Host ("desktop agora: {0} MiB  ->  orcamento para o llama: {1} MiB`n" -f $desktop, $budget) -ForegroundColor Cyan

function Split-Modelo($m) {
    $o = & "$bin\llama-fit-params.exe" -m $m -c 32768 -ctk q4_0 -ctv q4_0 -fa on --fit-print on 2>$null
    $cuda = ($o | Select-String '^CUDA0').ToString() -split '\s+'
    $host_ = ($o | Select-String '^Host').ToString()  -split '\s+'
    [pscustomobject]@{ cuda0 = [int]$cuda[1]; host = [int]$host_[1] }
}

function Teto($m, $ctk) {
    # varre contextos e devolve o maior que cabe no orcamento
    $melhor = 0
    foreach ($c in 131072, 262144, 393216, 524288, 655360, 786432, 1048576) {
        $o = & "$bin\llama-fit-params.exe" -m $m -c $c -ctk $ctk -ctv $ctk -fa on -b 512 -ub 512 --fit-print on 2>$null
        $p = ($o | Select-String '^CUDA0').ToString() -split '\s+'
        $total = [int]$p[1] + [int]$p[2] + [int]$p[3]
        if ($total -le $budget) { $melhor = $c } else { break }
    }
    return $melhor
}

foreach ($m in @($Model, $Ref)) {
    if (-not (Test-Path $m.Replace('/', '\'))) { Write-Host "faltando: $m" -ForegroundColor Yellow; continue }
    $nome = Split-Path $m -Leaf
    $s = Split-Modelo $m
    Write-Host ("{0}" -f $nome) -ForegroundColor Green
    Write-Host ("  CUDA0 {0} MiB | host {1} MiB" -f $s.cuda0, $s.host)
    foreach ($t in 'q4_0', 'q8_0') {
        Write-Host ("  teto de contexto com KV {0,-5}: {1:N0} tokens" -f $t, (Teto $m $t))
    }
}

if ($SkipPpl) { return }

Write-Host "`nperplexidade (ctx $PplCtx, $Chunks chunks) - so faz sentido comparar no mesmo corpus" -ForegroundColor Cyan
foreach ($m in @($Model, $Ref)) {
    if (-not (Test-Path $m.Replace('/', '\'))) { continue }
    $nome = Split-Path $m -Leaf
    $log = "E:\DEV\turboquant\logs\ppl-$($nome -replace '\.gguf$','').log"
    & "$bin\llama-perplexity.exe" -m $m -f $Corpus -c $PplCtx --chunks $Chunks `
        -ctk q4_0 -ctv q4_0 -fa on -ngl 99 -fit off -b 512 -ub 512 --log-file $log 2>&1 | Out-Null
    $t = Get-Content $log -Raw -ErrorAction SilentlyContinue
    if ($t -match 'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)') {
        Write-Host ("  {0,-34} PPL = {1} +/- {2}" -f $nome, $matches[1], $matches[2])
    } else {
        Write-Host ("  {0,-34} falhou, ver {1}" -f $nome, $log)
    }
}
