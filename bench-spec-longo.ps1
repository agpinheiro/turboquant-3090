# Bench de speculative decoding a CONTEXTO CHEIO - a confirmacao que importa.
#
# O bench-spec.ps1 mede a 12k de prompt. Mas a dor real esta a contexto cheio, onde a geracao
# cai de 60 para ~30 t/s porque a atencao das 16 camadas com KV passa a pesar no passo. Nada
# garante que o vencedor a 12k continue vencendo a 100k:
#   - o n-gram ganha um palheiro muito maior para casar (deveria melhorar), mas a busca custa mais;
#   - o MTP tem aceitacao estavel com contexto (74% medido a 166k), entao o ganho relativo do
#     n-gram sobre ele pode mudar de sinal.
#
# Cada variante: sobe o server, enche o contexto UMA vez, e depois roda os turnos reaproveitando
# o prefixo via cache_prompt. Assim o prefill de ~2.5 min e pago so uma vez por variante.
#
# Uso:
#   .\bench-spec-longo.ps1                              os 3 finalistas padrao
#   .\bench-spec-longo.ps1 -Only mtp2,mtp2-ngram        escolhidos a mao
#   .\bench-spec-longo.ps1 -PromptChars 250000          contexto menor, roda mais rapido
#   .\bench-spec-longo.ps1 -DryRun
#
# SEGURANCA: mesma regra do bench-spec.ps1 - aborta a variante se sobrar menos que -MinFree MiB.

param(
    # os finalistas: baseline atual, H1 puro, H1+H3 combinados
    [string[]] $Only         = @("mtp2", "mtp2-ngram", "mtp4-ngram-p60"),
    [int]      $Ctx          = 131072,
    [int]      $PromptChars  = 420000,   # ~100k tokens em warpeace-700k.txt (~4.2 chars/token)
    [int]      $Port         = 8099,
    [string]   $Model        = "E:/models/Qwen3.8-27B-Q4_K_M.gguf",
    [string]   $Kv           = "q4_0",
    [int]      $MinFree      = 1200,
    [int]      $LoadTimeout  = 420,
    [string]   $OutDir       = "E:\DEV\turboquant\logs\bench-spec-longo",
    [switch]   $DryRun
)

$ErrorActionPreference = "Stop"

# $VARIANTS + helpers, compartilhados com bench-spec.ps1
. (Join-Path $PSScriptRoot "bench-spec-common.ps1")

# ---------------------------------------------------------------- turnos
#
# Todos compartilham o mesmo prefixo gigante, entao do 2o em diante o prefill e quase zero
# (cache_prompt). O que sobra medido e geracao pura a contexto cheio, que e o alvo.

function Get-Turnos([string]$Prefixo) {
    @(
        # nao medido: so estabelece o cache do prefixo
        @{
            id      = "warmup"
            max_tok = 8
            medir   = $false
            prompt  = $Prefixo + "`n`n-----`nResponda apenas: ok."
        }

        # saida nova, o n-gram nao tem o que casar mesmo com 100k de palheiro
        @{
            id      = "novo"
            max_tok = 250
            medir   = $true
            prompt  = $Prefixo + "`n`n-----`nEscreva em portugues uma analise original de 200 palavras sobre a estrutura social retratada no texto acima. Nao cite trechos, nao copie frases."
        }

        # saida com muita copia literal de um contexto enorme: o melhor caso do n-gram
        @{
            id      = "citar"
            max_tok = 400
            medir   = $true
            prompt  = $Prefixo + "`n`n-----`nExtraia do texto acima 6 trechos literais longos (copiados exatamente, entre aspas, pelo menos 25 palavras cada) que envolvam Napoleao ou a guerra. So os trechos, um por linha."
        }

        # caso Claude Code: arquivo colado no fim de um contexto ja cheio, devolvido editado
        @{
            id      = "codigo"
            max_tok = 700
            medir   = $true
            prompt  = $Prefixo + "`n`n-----`nIgnore o texto acima. Aqui esta um arquivo:`n`n" +
                      (Get-Content -Raw -Path (Join-Path $script:BENCH_DATA "bench-code.txt")) +
                      "`n`n-----`nReescreva o arquivo INTEIRO, do inicio ao fim, mudando so o valor de MODEL para E:/models/Qwen3.8-27B-Q3_K_M.gguf. Preserve todo o resto exatamente, inclusive comentarios. Responda so com o arquivo."
        }
    )
}

function New-Row($Variante, $Turno, $Status, $Free) {
    return [pscustomobject]@{
        variante = $Variante; turno = $Turno; status = $Status
        gen_ts = $null; prompt_ts = $null; prompt_n = $null; cache_n = $null
        draft_acc = $null; draft_n = $null; draft_ok = $null; gen_tok = $null; free_mib = $Free
    }
}

# ---------------------------------------------------------------- execucao

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

$todo = $VARIANTS | Where-Object { $Only -contains $_.id }
if (@($todo).Count -eq 0) {
    $nomes = ($VARIANTS | ForEach-Object { $_.id }) -join ", "
    Write-Error "nenhuma variante casou com -Only. Disponiveis: $nomes"
    return
}

$fonte = Join-Path $script:BENCH_DATA "warpeace-700k.txt"
$texto = Get-Content -Raw -Path $fonte
if ($texto.Length -lt $PromptChars) {
    Write-Error "$fonte tem so $($texto.Length) chars, menos que -PromptChars $PromptChars"
    return
}
$prefixo = $texto.Substring(0, $PromptChars)
$turnos  = Get-Turnos $prefixo

Write-Host ""
Write-Host "  modelo   : $Model" -ForegroundColor Cyan
Write-Host "  ctx      : $Ctx   KV $Kv   porta $Port" -ForegroundColor Cyan
Write-Host "  prefixo  : $PromptChars chars (~$([math]::Round($PromptChars/4.2/1000))k tokens) de $fonte" -ForegroundColor Cyan
Write-Host "  variantes: $(@($todo).Count)   turnos medidos: $(($turnos | Where-Object { $_.medir }).Count)" -ForegroundColor Cyan
Write-Host "  saida    : $OutDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "  o primeiro turno de cada variante paga o prefill inteiro (~2-3 min). Os demais" -ForegroundColor DarkGray
Write-Host "  reaproveitam o prefixo via cache_prompt e medem geracao pura." -ForegroundColor DarkGray
Write-Host ""

$results = @()

foreach ($v in $todo) {
    $log     = Join-Path $OutDir ("server-" + $v.id + ".log")
    $allArgs = New-ServerArgs -SpecArgs $v.args -Model $Model -Ctx $Ctx -Kv $Kv -Port $Port -LogPath $log

    Write-Host ("=== {0}  -  {1}" -f $v.id, $v.desc) -ForegroundColor Yellow

    if ($DryRun) {
        Write-Host ("    " + $script:BENCH_BIN + " " + ($allArgs -join " ")) -ForegroundColor DarkGray
        continue
    }

    Remove-Item $log -ErrorAction SilentlyContinue
    $up = Start-BenchServer -AllArgs $allArgs -Port $Port -TimeoutSec $LoadTimeout -MinFree $MinFree

    if ($up.status -ne "ok") {
        if ($up.status -eq "vram-unsafe") {
            Write-Host ("    ABORTADA: so {0} MiB livres (minimo {1}). Reduza -Ctx." -f $up.free, $MinFree) -ForegroundColor Red
        }
        else {
            Write-Host ("    FALHOU ao subir ({0}) - ver {1}" -f $up.status, $log) -ForegroundColor Red
        }
        $results += New-Row $v.id "-" $up.status $up.free
        Stop-AllServers
        continue
    }

    Write-Host ("    carregou, {0} MiB livres" -f $up.free) -ForegroundColor DarkGray

    foreach ($tn in $turnos) {
        try {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $r  = Invoke-Chat -Port $Port -Prompt $tn.prompt -MaxTok $tn.max_tok
            $sw.Stop()
            $t = $r.timings

            if (-not $tn.medir) {
                Write-Host ("    {0,-8} prefill {1,7:N0} tokens a {2,6:N0} t/s  ({3:N0}s)" -f `
                    $tn.id, $t.prompt_n, $t.prompt_per_second, $sw.Elapsed.TotalSeconds) -ForegroundColor DarkGray
                continue
            }

            $acc = $null
            if ($t.draft_n -gt 0) { $acc = [math]::Round($t.draft_n_accepted / $t.draft_n, 4) }

            $row = [pscustomobject]@{
                variante  = $v.id
                turno     = $tn.id
                status    = "ok"
                gen_ts    = [math]::Round($t.predicted_per_second, 2)
                prompt_ts = [math]::Round($t.prompt_per_second, 1)
                prompt_n  = $t.prompt_n
                cache_n   = $t.cache_n
                draft_acc = $acc
                draft_n   = $t.draft_n
                draft_ok  = $t.draft_n_accepted
                gen_tok   = $t.predicted_n
                free_mib  = $up.free
            }
            $results += $row
            Write-Host ("    {0,-8} {1,7:N2} t/s   draft {2}/{3}   cache {4:N0}, reprefill {5:N0}" -f `
                $tn.id, $row.gen_ts, $t.draft_n_accepted, $t.draft_n, $t.cache_n, $t.prompt_n)

            $r.choices[0].message.content | Out-File -Encoding utf8 `
                -FilePath (Join-Path $OutDir ("resp-" + $v.id + "-" + $tn.id + ".txt"))
        }
        catch {
            Write-Host ("    {0,-8} ERRO: {1}" -f $tn.id, $_.Exception.Message) -ForegroundColor Red
            $results += New-Row $v.id $tn.id "erro" $up.free
        }
    }

    Stop-AllServers
    Write-Host ""
}

if ($DryRun) { return }

# ---------------------------------------------------------------- relatorio

$csv = Join-Path $OutDir "resultados.csv"
$results | Export-Csv -Path $csv -NoTypeInformation -Encoding UTF8
Write-Host "CSV: $csv" -ForegroundColor Cyan
Write-Host ""

$ok = $results | Where-Object { $_.status -eq "ok" }
if ($ok.Count -eq 0) {
    Write-Warning "nenhuma medida valida"
    return
}

$base = @{}
foreach ($r in ($ok | Where-Object { $_.variante -eq "mtp2" })) { $base[$r.turno] = $r.gen_ts }

$ids = @("novo", "citar", "codigo")
$fmt = "{0,-16} {1,20} {2,20} {3,20}"

Write-Host "=== t/s de geracao a contexto cheio (ganho vs mtp2) ===" -ForegroundColor Green
Write-Host ($fmt -f "variante", "novo", "citar", "codigo")
foreach ($v in $todo) {
    $cells = @()
    foreach ($tid in $ids) {
        $r = $ok | Where-Object { $_.variante -eq $v.id -and $_.turno -eq $tid } | Select-Object -First 1
        if ($null -eq $r) { $cells += "-"; continue }
        $cell = "{0,7:N2}" -f $r.gen_ts
        if ($v.id -ne "mtp2" -and $base.ContainsKey($tid) -and $base[$tid] -gt 0) {
            $cell += "  ({0,6:P0})" -f (($r.gen_ts / $base[$tid]) - 1)
        }
        $cells += $cell
    }
    Write-Host ($fmt -f $v.id, $cells[0], $cells[1], $cells[2])
}

Write-Host ""
Write-Host "=== aceitacao de draft (aceitos/rascunhados) ===" -ForegroundColor Green
Write-Host ($fmt -f "variante", "novo", "citar", "codigo")
foreach ($v in $todo) {
    $cells = @()
    foreach ($tid in $ids) {
        $r = $ok | Where-Object { $_.variante -eq $v.id -and $_.turno -eq $tid } | Select-Object -First 1
        if ($null -eq $r -or $null -eq $r.draft_acc) { $cells += "-"; continue }
        $cells += "{0,6:P1}  {1}/{2}" -f $r.draft_acc, $r.draft_ok, $r.draft_n
    }
    Write-Host ($fmt -f $v.id, $cells[0], $cells[1], $cells[2])
}

Write-Host ""
Write-Host "Se 'reprefill' nao for pequeno nos turnos medidos, o cache_prompt nao pegou e os" -ForegroundColor DarkGray
Write-Host "numeros de t/s de geracao continuam validos, mas o teste demorou a toa." -ForegroundColor DarkGray
