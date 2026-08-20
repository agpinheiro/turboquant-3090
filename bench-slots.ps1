# Bench da dimensao -np: vale a pena abrir 2 slots em vez de 1?
#
# Slot unico mede LATENCIA (t/s de um stream). Varios slots medem THROUGHPUT AGREGADO: com 2
# requisicoes no mesmo forward pass, a leitura dos pesos (o termo dominante, ~28 ms de um passo
# de ~31 ms) e amortizada entre as duas. O t/s por stream cai um pouco, o agregado sobe muito.
#
# Hipotese secundaria, e a mais interessante aqui: speculative decoding e batching atacam o MESMO
# gargalo - amortizar a leitura dos pesos. Com -np 2 o ganho do MTP/n-gram deveria ENCOLHER,
# porque o batch ja fez parte do trabalho. Por isso o 'nospec' entra como controle.
#
# Metodo: em ambas as configuracoes as MESMAS N requisicoes sao disparadas simultaneamente.
# Com -np 1 o server serializa; com -np 2 ele agrupa. O que se compara e o relogio de parede
# ate a ultima terminar - que e o que o usuario sente.
#
# Uso:
#   .\bench-slots.ps1                                  nospec/mtp2/mtp2-ngram x 1 e 2 slots
#   .\bench-slots.ps1 -Slots 1,2,4                     inclui 4 slots
#   .\bench-slots.ps1 -Only mtp2 -Slots 1,2
#   .\bench-slots.ps1 -DryRun
#
# NOTA sobre o contexto: -c e o KV TOTAL, dividido entre os slots. Com -c 131072 -np 2 cada slot
# fica com 65536. Isso e deliberado: mantem a VRAM constante entre as configuracoes, que e o
# controle certo numa placa que ja opera no limite. O script imprime o n_ctx_slot efetivo.
#
# SEGURANCA: mesma regra dos outros - aborta a variante se sobrar menos que -MinFree MiB.
# Atencao: mais slots custam mais VRAM quando ha MTP (cada slot tem seu contexto de draft).

param(
    [string[]] $Only         = @("nospec", "mtp2", "mtp2-ngram"),
    [int[]]    $Slots        = @(1, 2),
    [int]      $Batch        = 0,          # requisicoes simultaneas; 0 = usar max($Slots)
    [int]      $Ctx          = 131072,
    [int]      $Port         = 8099,
    [string]   $Model        = "E:/models/Qwen3.8-27B-Q4_K_M.gguf",
    [string]   $Kv           = "q4_0",
    [int]      $MinFree      = 1200,
    [int]      $LoadTimeout  = 420,
    # "off" desliga o thinking: sem isso o modelo gasta o max_tokens inteiro raciocinando e
    # message.content sai vazio (ver logs/bench-spec-run1.txt). "auto" mede o regime com thinking.
    [string]   $Reasoning    = "off",
    [string]   $OutDir       = "E:\DEV\turboquant\logs\bench-slots",
    [switch]   $DryRun
)

$ErrorActionPreference = "Stop"

# $VARIANTS + helpers, compartilhados com bench-spec.ps1
. (Join-Path $PSScriptRoot "bench-spec-common.ps1")

if ($Batch -le 0) { $Batch = ($Slots | Measure-Object -Maximum).Maximum }

# ---------------------------------------------------------------- cenarios
#
# Os prompts de um cenario sao DISTINTOS entre si de proposito: prompts iguais compartilhariam
# cache e o -np 1 sairia artificialmente bem. Por isso tambem cache_prompt = false.
# Prompts curtos e geracao longa, para que o decode domine - e o decode que o batching acelera.

function Get-Cenarios([int]$N) {
    $prosa = Get-Content -Raw -Path (Join-Path $script:BENCH_DATA "warpeace.txt")
    $code  = Get-Content -Raw -Path (Join-Path $script:BENCH_DATA "bench-code.txt")

    $temas = @(
        "a relacao entre poder militar e vida de salao",
        "o papel das mulheres nas conversas politicas",
        "a distancia entre a nobreza e o povo",
        "o uso do frances como marca de classe"
    )

    # cada requisicao pega uma fatia diferente do texto e um tema diferente
    $gerar = @()
    for ($i = 0; $i -lt $N; $i++) {
        $ini = 5000 + ($i * 9000)
        $gerar += ($prosa.Substring($ini, 4000) +
            "`n`n-----`nCom base no trecho acima, escreva em portugues uma analise original de 300 palavras sobre " +
            $temas[$i % $temas.Count] + ". Nao copie frases do texto.")
    }

    # cada requisicao reescreve o mesmo arquivo com uma mudanca diferente:
    # saida quase toda repetida do prompt, que e onde o n-gram trabalha
    $alvos = @(
        "o valor de MODEL passa a ser E:/models/Qwen3.8-27B-Q3_K_M.gguf",
        "o CTX padrao passa de 180000 para 160000",
        "a porta padrao passa de 8080 para 9090",
        "o ALIAS padrao passa de qwen3.8-27b para qwen-local"
    )
    $codigo = @()
    for ($i = 0; $i -lt $N; $i++) {
        $codigo += ("Aqui esta um arquivo:`n`n" + $code +
            "`n`n-----`nReescreva o arquivo INTEIRO, do inicio ao fim, mudando apenas isto: " +
            $alvos[$i % $alvos.Count] +
            ". Preserve todo o resto exatamente, inclusive comentarios. Responda so com o arquivo.")
    }

    @(
        @{ id = "gerar";  max_tok = 400; prompts = $gerar }
        @{ id = "codigo"; max_tok = 800; prompts = $codigo }
    )
}

function Get-SlotCtx([string]$LogPath) {
    if (-not (Test-Path $LogPath)) { return $null }
    $m = Select-String -Path $LogPath -Pattern "n_ctx_slot\s*=\s*(\d+)" | Select-Object -First 1
    if ($null -eq $m) { return $null }
    return [int]$m.Matches[0].Groups[1].Value
}

function New-Row($Variante, $Np, $Cenario, $Status, $Free) {
    return [pscustomobject]@{
        variante = $Variante; np = $Np; cenario = $Cenario; status = $Status
        wall_s = $null; agregado_ts = $null; por_stream_ts = $null
        gen_tok = $null; draft_acc = $null; draft_n = $null; draft_ok = $null
        slot_ctx = $null; free_mib = $Free
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

$cenarios = Get-Cenarios $Batch

Write-Host ""
Write-Host "  modelo    : $Model" -ForegroundColor Cyan
Write-Host "  ctx TOTAL : $Ctx   KV $Kv   porta $Port" -ForegroundColor Cyan
Write-Host "  slots     : $($Slots -join ', ')   requisicoes simultaneas: $Batch" -ForegroundColor Cyan
Write-Host "  variantes : $(@($todo).Count)   cenarios: $($cenarios.Count)" -ForegroundColor Cyan
Write-Host "  saida     : $OutDir" -ForegroundColor Cyan
Write-Host ""

$results = @()

foreach ($v in $todo) {
    foreach ($np in $Slots) {
        $tag     = "{0}-np{1}" -f $v.id, $np
        $log     = Join-Path $OutDir ("server-" + $tag + ".log")
        $allArgs = New-ServerArgs -SpecArgs $v.args -Model $Model -Ctx $Ctx -Kv $Kv `
                                  -Port $Port -LogPath $log -Slots $np -Reasoning $Reasoning

        Write-Host ("=== {0}  (np={1})  -  {2}" -f $v.id, $np, $v.desc) -ForegroundColor Yellow

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
            $results += New-Row $v.id $np "-" $up.status $up.free
            Stop-AllServers
            continue
        }

        $slotCtx = Get-SlotCtx $log
        Write-Host ("    carregou, {0} MiB livres, n_ctx_slot = {1}" -f $up.free, $slotCtx) -ForegroundColor DarkGray

        foreach ($c in $cenarios) {
            try {
                $res = Invoke-ChatParallel -Port $Port -Prompts $c.prompts -MaxTok $c.max_tok -CachePrompt $false

                $genTot   = 0
                $draftN   = 0
                $draftOk  = 0
                $streamTs = @()
                foreach ($r in $res.responses) {
                    $t = $r.timings
                    $genTot   += $t.predicted_n
                    $draftN   += $t.draft_n
                    $draftOk  += $t.draft_n_accepted
                    $streamTs += $t.predicted_per_second
                }

                $acc = $null
                if ($draftN -gt 0) { $acc = [math]::Round($draftOk / $draftN, 4) }

                $row = [pscustomobject]@{
                    variante      = $v.id
                    np            = $np
                    cenario       = $c.id
                    status        = "ok"
                    wall_s        = [math]::Round($res.wall, 2)
                    # o numero que importa: tokens gerados no total / relogio de parede
                    agregado_ts   = [math]::Round($genTot / $res.wall, 2)
                    por_stream_ts = [math]::Round((($streamTs | Measure-Object -Average).Average), 2)
                    gen_tok       = $genTot
                    draft_acc     = $acc
                    draft_n       = $draftN
                    draft_ok      = $draftOk
                    slot_ctx      = $slotCtx
                    free_mib      = $up.free
                }
                $results += $row
                Write-Host ("    {0,-7} agregado {1,7:N2} t/s   por stream {2,6:N2} t/s   {3:N1}s   draft {4}/{5}" -f `
                    $c.id, $row.agregado_ts, $row.por_stream_ts, $row.wall_s, $draftOk, $draftN)

                # Save-Response avisa se a saida veio de reasoning_content em vez de content,
                # que e o sintoma de thinking ligado comendo o max_tokens inteiro
                $origens = @()
                for ($i = 0; $i -lt $res.responses.Count; $i++) {
                    $origens += Save-Response $res.responses[$i] `
                        (Join-Path $OutDir ("resp-" + $tag + "-" + $c.id + "-" + $i + ".txt"))
                }
                $ruins = @($origens | Where-Object { $_ -ne "content" })
                if ($ruins.Count -gt 0) {
                    Write-Host ("            aviso: {0}/{1} respostas vieram de '{2}', nao de content" -f `
                        $ruins.Count, $origens.Count, $ruins[0]) -ForegroundColor Red
                }
            }
            catch {
                Write-Host ("    {0,-7} ERRO: {1}" -f $c.id, $_.Exception.Message) -ForegroundColor Red
                $results += New-Row $v.id $np $c.id "erro" $up.free
            }
        }

        Stop-AllServers
        Write-Host ""
    }
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

$fmt = "{0,-16} {1,4} {2,24} {3,24}"

Write-Host "=== throughput AGREGADO (todos os streams somados) ===" -ForegroundColor Green
Write-Host ($fmt -f "variante", "np", "gerar", "codigo")
foreach ($v in $todo) {
    foreach ($np in $Slots) {
        $cells = @()
        foreach ($cid in @("gerar", "codigo")) {
            $r = $ok | Where-Object { $_.variante -eq $v.id -and $_.np -eq $np -and $_.cenario -eq $cid } | Select-Object -First 1
            if ($null -eq $r) { $cells += "-"; continue }
            $cell = "{0,7:N2} t/s" -f $r.agregado_ts
            # ganho do np atual sobre o np=1 da MESMA variante: isola o efeito do batching
            $b = $ok | Where-Object { $_.variante -eq $v.id -and $_.np -eq 1 -and $_.cenario -eq $cid } | Select-Object -First 1
            if ($np -ne 1 -and $null -ne $b -and $b.agregado_ts -gt 0) {
                $cell += "  ({0,6:P0})" -f (($r.agregado_ts / $b.agregado_ts) - 1)
            }
            $cells += $cell
        }
        Write-Host ($fmt -f $v.id, $np, $cells[0], $cells[1])
    }
}

Write-Host ""
Write-Host "=== t/s por stream (a latencia que o usuario sente) ===" -ForegroundColor Green
Write-Host ($fmt -f "variante", "np", "gerar", "codigo")
foreach ($v in $todo) {
    foreach ($np in $Slots) {
        $cells = @()
        foreach ($cid in @("gerar", "codigo")) {
            $r = $ok | Where-Object { $_.variante -eq $v.id -and $_.np -eq $np -and $_.cenario -eq $cid } | Select-Object -First 1
            if ($null -eq $r) { $cells += "-"; continue }
            $cells += "{0,7:N2} t/s" -f $r.por_stream_ts
        }
        Write-Host ($fmt -f $v.id, $np, $cells[0], $cells[1])
    }
}

Write-Host ""
Write-Host "=== VRAM e contexto por slot ===" -ForegroundColor Green
Write-Host ("{0,-16} {1,4} {2,12} {3,12}" -f "variante", "np", "n_ctx_slot", "livre MiB")
foreach ($v in $todo) {
    foreach ($np in $Slots) {
        $r = $ok | Where-Object { $_.variante -eq $v.id -and $_.np -eq $np } | Select-Object -First 1
        if ($null -eq $r) { continue }
        Write-Host ("{0,-16} {1,4} {2,12} {3,12}" -f $v.id, $np, $r.slot_ctx, $r.free_mib)
    }
}

Write-Host ""
Write-Host "Leitura: se o agregado do np=2 subir bem menos no 'mtp2-ngram' do que no 'nospec'," -ForegroundColor DarkGray
Write-Host "batching e speculative estao brigando pelo mesmo gargalo - e escolher um so basta." -ForegroundColor DarkGray
