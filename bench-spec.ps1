# Bench de speculative decoding a contexto curto: t/s por variante de --spec-type / -n-max /
# --spec-draft-p-min. Testa H1 (n-gram junto com MTP) e H3 (p_min adaptativo), mais o sweep
# de n_max 2/3/4.
#
# Cada variante sobe um llama-server proprio, roda as mesmas 3 cargas, e morre. Com tudo o mais
# constante, o CSV final compara t/s e aceitacao de draft variante a variante.
#
# Uso:
#   .\bench-spec.ps1                          matriz inteira (13 variantes, ~40 min)
#   .\bench-spec.ps1 -Only mtp2,mtp2-ngram    so essas variantes
#   .\bench-spec.ps1 -DryRun                  imprime os comandos, nao carrega nada
#   .\bench-spec.ps1 -Ctx 65536               contexto menor, mais folga de VRAM
#
# Depois deste, rodar bench-spec-longo.ps1 com os finalistas: a dor real e a contexto cheio,
# e nada garante que o vencedor a 12k continue vencendo a 100k.
#
# SEGURANCA: o ctx padrao e 131072, nao 200k. Com 200k + MTP + Q4_K_M sobram ~980 MiB, o WDDM
# comeca a paginar VRAM para a RAM e a maquina trava sem escrever erro nenhum no log. O script
# aborta a variante se sobrar menos que -MinFree MiB depois de carregar.

param(
    [string[]] $Only        = @(),
    [int]      $Ctx         = 131072,
    [int]      $Port        = 8099,
    [string]   $Model       = "E:/models/Qwen3.8-27B-Q4_K_M.gguf",
    [string]   $Kv          = "q4_0",
    [int]      $MinFree     = 1200,
    [int]      $LoadTimeout = 420,
    # "off" desliga o thinking (padrao: e o unico jeito das cargas produzirem o artefato
    # pedido). "auto" ou "on" = thinking ligado, que e como o Claude Code de fato usa o modelo -
    # regime valido, mas ai o que se mede e geracao de tokens de raciocinio.
    [string]   $Reasoning   = "off",
    [string]   $OutDir      = "E:\DEV\turboquant\logs\bench-spec",
    [switch]   $DryRun
)

$ErrorActionPreference = "Stop"

# $VARIANTS + helpers (Start-BenchServer, New-ServerArgs, Invoke-Chat, ...)
. (Join-Path $PSScriptRoot "bench-spec-common.ps1")

# ---------------------------------------------------------------- cargas
#
# As tres cargas separam o caso em que o n-gram tem o que casar do caso em que nao tem.
# Sem essa separacao o resultado do H1 fica mascarado pela media.

function Get-Workloads {
    $code  = Get-Content -Raw -Path (Join-Path $script:BENCH_DATA "bench-code.txt")
    $prosa = (Get-Content -Raw -Path (Join-Path $script:BENCH_DATA "warpeace.txt")).Substring(0, 12000)

    @(
        # Saida majoritariamente NOVA: o n-gram nao tem o que casar.
        # Serve para provar que ligar n-gram nao custa nada quando ele erra.
        @{
            id      = "prosa"
            max_tok = 300
            prompt  = $prosa + "`n`n-----`nCom base no trecho acima, escreva uma analise original em portugues sobre o tom social da cena. Nao cite o texto, nao repita frases dele. Entre 200 e 300 palavras."
        }

        # Saida quase toda REPETIDA do prompt: e o caso Claude Code (le arquivo, devolve editado).
        # E aqui que o n-gram deveria explodir de ganho.
        @{
            id      = "codigo"
            max_tok = 900
            prompt  = "Aqui esta um arquivo:`n`n" + $code + "`n`n-----`nReescreva o arquivo INTEIRO, completo, do inicio ao fim, mudando apenas duas coisas: o valor de MODEL passa a ser E:/models/Qwen3.8-27B-Q3_K_M.gguf, e o CTX padrao passa de 180000 para 160000. Preserve todo o resto exatamente, inclusive os comentarios. Responda so com o arquivo."
        }

        # Meio-termo: trechos literais intercalados com texto novo. Tipico de RAG / agente.
        @{
            id      = "citacao"
            max_tok = 400
            prompt  = $prosa + "`n`n-----`nExtraia do texto acima 5 trechos literais (copiados exatamente, entre aspas) que mostrem o clima politico da conversa. Antes de cada trecho, escreva uma frase curta explicando por que ele importa."
        }
    )
}

function New-Row($Variante, $Carga, $Status, $Free) {
    return [pscustomobject]@{
        variante = $Variante; carga = $Carga; status = $Status; origem = $null
        gen_ts = $null; prompt_ts = $null; draft_acc = $null
        draft_n = $null; draft_ok = $null; gen_tok = $null; free_mib = $Free
    }
}

# ---------------------------------------------------------------- execucao

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

$todo = $VARIANTS
if ($Only.Count -gt 0) {
    $todo = $VARIANTS | Where-Object { $Only -contains $_.id }
    if (@($todo).Count -eq 0) {
        $nomes = ($VARIANTS | ForEach-Object { $_.id }) -join ", "
        Write-Error "nenhuma variante casou com -Only. Disponiveis: $nomes"
        return
    }
}

$workloads = Get-Workloads

Write-Host ""
Write-Host "  modelo   : $Model" -ForegroundColor Cyan
Write-Host "  ctx      : $Ctx   KV $Kv   porta $Port" -ForegroundColor Cyan
Write-Host "  variantes: $(@($todo).Count)   cargas: $($workloads.Count)" -ForegroundColor Cyan
Write-Host "  saida    : $OutDir" -ForegroundColor Cyan
Write-Host ""

$results = @()

foreach ($v in $todo) {
    $log     = Join-Path $OutDir ("server-" + $v.id + ".log")
    $allArgs = New-ServerArgs -SpecArgs $v.args -Model $Model -Ctx $Ctx -Kv $Kv -Port $Port `
                              -LogPath $log -Reasoning $Reasoning

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

    foreach ($w in $workloads) {
        try {
            $r = Invoke-Chat -Port $Port -Prompt $w.prompt -MaxTok $w.max_tok
            $t = $r.timings

            $acc = $null
            if ($t.draft_n -gt 0) { $acc = [math]::Round($t.draft_n_accepted / $t.draft_n, 4) }

            # guarda a resposta e registra de onde veio: o 'codigo' tem que sair correto, nao so
            # rapido, e "reasoning" significa que a carga mediu raciocinio, nao o artefato pedido
            $origem = Save-Response $r (Join-Path $OutDir ("resp-" + $v.id + "-" + $w.id + ".txt"))

            $row = [pscustomobject]@{
                variante  = $v.id
                carga     = $w.id
                status    = "ok"
                origem    = $origem
                gen_ts    = [math]::Round($t.predicted_per_second, 2)
                prompt_ts = [math]::Round($t.prompt_per_second, 1)
                draft_acc = $acc
                draft_n   = $t.draft_n
                draft_ok  = $t.draft_n_accepted
                gen_tok   = $t.predicted_n
                free_mib  = $up.free
            }
            $results += $row

            $aviso = ""
            if ($origem -ne "content") { $aviso = "   <-- saida em '$origem', nao em content!" }
            Write-Host ("    {0,-8} {1,7:N2} t/s   draft {2}/{3}{4}" -f `
                $w.id, $row.gen_ts, $t.draft_n_accepted, $t.draft_n, $aviso)
        }
        catch {
            Write-Host ("    {0,-8} ERRO: {1}" -f $w.id, $_.Exception.Message) -ForegroundColor Red
            $results += New-Row $v.id $w.id "erro" $up.free
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

# t/s por carga, com o ganho sobre o mtp2 (a config em producao hoje)
$base = @{}
foreach ($r in ($ok | Where-Object { $_.variante -eq "mtp2" })) { $base[$r.carga] = $r.gen_ts }

$cargas = @("prosa", "codigo", "citacao")
$fmt    = "{0,-16} {1,20} {2,20} {3,20}"

Write-Host "=== t/s de geracao por carga (ganho vs mtp2) ===" -ForegroundColor Green
Write-Host ($fmt -f "variante", "prosa", "codigo", "citacao")
foreach ($v in $todo) {
    $cells = @()
    foreach ($wid in $cargas) {
        $r = $ok | Where-Object { $_.variante -eq $v.id -and $_.carga -eq $wid } | Select-Object -First 1
        if ($null -eq $r) { $cells += "-"; continue }
        $cell = "{0,7:N2}" -f $r.gen_ts
        if ($v.id -ne "mtp2" -and $base.ContainsKey($wid) -and $base[$wid] -gt 0) {
            $cell += "  ({0,6:P0})" -f (($r.gen_ts / $base[$wid]) - 1)
        }
        $cells += $cell
    }
    Write-Host ($fmt -f $v.id, $cells[0], $cells[1], $cells[2])
}

Write-Host ""
Write-Host "=== aceitacao de draft (aceitos/rascunhados) ===" -ForegroundColor Green
Write-Host ($fmt -f "variante", "prosa", "codigo", "citacao")
foreach ($v in $todo) {
    $cells = @()
    foreach ($wid in $cargas) {
        $r = $ok | Where-Object { $_.variante -eq $v.id -and $_.carga -eq $wid } | Select-Object -First 1
        if ($null -eq $r -or $null -eq $r.draft_acc) { $cells += "-"; continue }
        $cells += "{0,6:P1}  {1}/{2}" -f $r.draft_acc, $r.draft_ok, $r.draft_n
    }
    Write-Host ($fmt -f $v.id, $cells[0], $cells[1], $cells[2])
}

Write-Host ""
Write-Host "Respostas em $OutDir\resp-<variante>-<carga>.txt - confira se o 'codigo' saiu correto." -ForegroundColor DarkGray
Write-Host "Proximo passo: .\bench-spec-longo.ps1 -Only <os 2-3 melhores>" -ForegroundColor DarkGray
