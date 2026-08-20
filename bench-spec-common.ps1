# Variantes de speculative decoding e helpers de bench, compartilhados por bench-spec.ps1
# (contexto curto, matriz completa) e bench-spec-longo.ps1 (contexto cheio, so os finalistas).
# Arquivo unico para os dois scripts nao divergirem.
#
# Cada variante muda SO os args de speculative. Modelo, ctx, KV e batch ficam constantes.
#
# Por que n=3 e n=4 nao sao redundantes com o que ja foi medido: bare n=3 deu 53.4 t/s contra
# 60.1 do n=2 porque com p_min=0 ele rascunha 3 tokens SEMPRE, mesmo sem confianca. Com p_min>0
# o comprimento vira adaptativo - e a celula n=4 + p_min que pode ganhar onde n=4 puro perde.
#
# Por que o n-gram nao e limitado pelo --spec-draft-n-max: o get_n_draft_max() do server
# (server-context.cpp:449) so limita por espaco de contexto; a flag vale so para os speculators
# com draft model. Os n-gram tem orcamento proprio - ngram_map.size_m = 48, ngram_mod.n_max = 64.
# E a prioridade em speculative.cpp:2472-2480 poe n-gram ANTES do MTP: o primeiro que produzir
# draft nao-vazio vence, entao o n-gram tenta primeiro (custo zero) e o MTP pega o resto.

$VARIANTS = @(
    @{ id = "nospec";         desc = "controle, sem speculative";                  args = @() }

    # --- sweep de n_max puro (p_min = 0: rascunha sempre o maximo) ---
    @{ id = "mtp2";           desc = "MTP n=2 (config atual, baseline)";           args = @("--spec-type", "draft-mtp", "--spec-draft-n-max", "2") }
    @{ id = "mtp3";           desc = "MTP n=3";                                    args = @("--spec-type", "draft-mtp", "--spec-draft-n-max", "3") }
    @{ id = "mtp4";           desc = "MTP n=4";                                    args = @("--spec-type", "draft-mtp", "--spec-draft-n-max", "4") }

    # --- H3: p_min corta o rascunho quando a cabeca nao tem confianca ---
    @{ id = "mtp2-p60";       desc = "MTP n=2 + p_min 0.6";                        args = @("--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.6") }
    @{ id = "mtp3-p60";       desc = "MTP n=3 + p_min 0.6";                        args = @("--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.6") }
    @{ id = "mtp4-p60";       desc = "MTP n=4 + p_min 0.6";                        args = @("--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--spec-draft-p-min", "0.6") }
    @{ id = "mtp4-p80";       desc = "MTP n=4 + p_min 0.8 (mais conservador)";     args = @("--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--spec-draft-p-min", "0.8") }

    # --- H1: n-gram, draft de custo zero, ate 48 tokens (size_m) ---
    @{ id = "ngram";          desc = "so ngram-map-k4v, sem MTP";                  args = @("--spec-type", "ngram-map-k4v") }
    @{ id = "ngrammod";       desc = "so ngram-mod, sem MTP";                      args = @("--spec-type", "ngram-mod") }
    @{ id = "mtp2-ngram";     desc = "H1: MTP n=2 + ngram-map-k4v";                args = @("--spec-type", "draft-mtp,ngram-map-k4v", "--spec-draft-n-max", "2") }
    @{ id = "mtp2-ngrammod";  desc = "H1: MTP n=2 + ngram-mod";                    args = @("--spec-type", "draft-mtp,ngram-mod", "--spec-draft-n-max", "2") }

    # --- H1 + H3: n-gram cobre a repeticao, MTP longo e adaptativo cobre o resto ---
    @{ id = "mtp4-ngram-p60"; desc = "H1+H3: MTP n=4 p_min 0.6 + ngram-map-k4v";   args = @("--spec-type", "draft-mtp,ngram-map-k4v", "--spec-draft-n-max", "4", "--spec-draft-p-min", "0.6") }
)

# ---------------------------------------------------------------- helpers de bench

$script:BENCH_BIN      = "E:\DEV\turboquant\llama.cpp\build\bin\Release\llama-server.exe"
$script:BENCH_TEMPLATE = "E:\DEV\turboquant\qwen35-tolerant.jinja"
$script:BENCH_DATA     = "E:\DEV\turboquant\data"

function Get-FreeVram {
    # @()[0] em vez de "| Select-Object -First 1": o -First quebra o pipeline no meio e mata o
    # nvidia-smi, deixando $LASTEXITCODE = -1. O resultado da medicao e o mesmo, mas o script
    # inteiro passava a sair com codigo 255 mesmo tendo terminado bem.
    $used = @(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits)[0].Trim()
    $tot  = @(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)[0].Trim()
    return ([int]$tot - [int]$used)
}

function Stop-AllServers {
    $procs = Get-Process llama-server -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Host "  parando llama-server ja no ar..." -ForegroundColor DarkGray
        $procs | Stop-Process -Force
    }
    # espera a VRAM voltar; sem isso a proxima variante tenta carregar em cima da anterior
    for ($i = 0; $i -lt 60; $i++) {
        if (-not (Get-Process llama-server -ErrorAction SilentlyContinue)) {
            if ((Get-FreeVram) -gt 8000) { return }
        }
        Start-Sleep -Milliseconds 1000
    }
    Write-Warning "VRAM nao voltou ao baseline depois de 60s"
}

function Wait-Health([int]$Port, [int]$TimeoutSec) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        if (-not (Get-Process llama-server -ErrorAction SilentlyContinue)) {
            return "morreu"
        }
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
            if ($h.status -eq "ok") { return "ok" }
        }
        catch { }
        Start-Sleep -Milliseconds 1500
    }
    return "timeout"
}

# Monta a linha de comando do server para uma variante. Tudo fora de $SpecArgs e constante
# entre variantes - e o que torna a comparacao valida.
function New-ServerArgs($SpecArgs, [string]$Model, [int]$Ctx, [string]$Kv, [int]$Port, [string]$LogPath, [int]$Slots = 1, [string]$Reasoning = "off") {
    $a = @(
        "-m", $Model,
        "-a", "bench",
        "--chat-template-file", $script:BENCH_TEMPLATE,
        "-c", "$Ctx",
        "-np", "$Slots",
        "-ctk", $Kv, "-ctv", $Kv,
        "-fa", "on",
        "-ngl", "99",
        "-fit", "off",
        "-b", "512", "-ub", "512",
        # -rea off passa enable_thinking=false para o template (o qwen35-tolerant.jinja suporta).
        # O default e 'auto', que detecta do template e liga o thinking: o modelo entao gasta o
        # max_tokens inteiro raciocinando, message.content sai VAZIO, e a carga nunca produz o
        # artefato que se queria medir. Ver logs/bench-spec-run1.txt e run2.
        # Cuidado: --reasoning-budget 0 NAO resolve isso - foi testado e e no-op aqui.
        "--reasoning", $Reasoning,
        "--host", "127.0.0.1", "--port", "$Port"
    )
    $s = @($SpecArgs)
    # speculators com draft model precisam do KV do contexto de draft quantizado tambem:
    # ele nasce em f16 (common/speculative.cpp) e come ~1.5 GiB a mais sem isso
    if (($s -join " ") -match "draft-mtp") {
        $s += @("-ctkd", $Kv, "-ctvd", $Kv)
    }
    return $a + $s + @("--log-file", $LogPath)
}

# Sobe o server e espera ficar saudavel. Devolve "ok", "morreu", "timeout" ou "vram-unsafe".
function Start-BenchServer($AllArgs, [int]$Port, [int]$TimeoutSec, [int]$MinFree) {
    Stop-AllServers
    Start-Process -FilePath $script:BENCH_BIN -ArgumentList $AllArgs -PassThru -WindowStyle Hidden | Out-Null

    $health = Wait-Health -Port $Port -TimeoutSec $TimeoutSec
    if ($health -ne "ok") { return @{ status = $health; free = $null } }

    $free = Get-FreeVram
    if ($free -lt $MinFree) { return @{ status = "vram-unsafe"; free = $free } }

    return @{ status = "ok"; free = $free }
}

function Invoke-Chat([int]$Port, [string]$Prompt, [int]$MaxTok, [bool]$CachePrompt = $true, [int]$TimeoutSec = 3600) {
    $payload = @{
        messages     = @(@{ role = "user"; content = $Prompt })
        max_tokens   = $MaxTok
        temperature  = 0
        seed         = 42
        stream       = $false
        cache_prompt = $CachePrompt
    }
    $json  = $payload | ConvertTo-Json -Depth 5 -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)

    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" `
        -Method Post -ContentType "application/json; charset=utf-8" `
        -Body $bytes -TimeoutSec $TimeoutSec
}

# Dispara N requisicoes SIMULTANEAS e espera todas. Sem isso o -np > 1 nao e exercitado:
# requisicoes sequenciais usam um slot so, por mais slots que o server tenha aberto.
# Usa HttpClient porque o Invoke-RestMethod do PS 5.1 e sincrono.
function Invoke-ChatParallel([int]$Port, [string[]]$Prompts, [int]$MaxTok, [bool]$CachePrompt = $false, [int]$TimeoutSec = 3600) {
    Add-Type -AssemblyName System.Net.Http -ErrorAction SilentlyContinue

    $client = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)
    $uri = "http://127.0.0.1:$Port/v1/chat/completions"

    $tasks = New-Object System.Collections.ArrayList
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    foreach ($p in $Prompts) {
        $payload = @{
            messages     = @(@{ role = "user"; content = $p })
            max_tokens   = $MaxTok
            temperature  = 0
            seed         = 42
            stream       = $false
            cache_prompt = $CachePrompt
        } | ConvertTo-Json -Depth 5 -Compress

        $content = New-Object System.Net.Http.StringContent($payload, [System.Text.Encoding]::UTF8, "application/json")
        [void]$tasks.Add($client.PostAsync($uri, $content))
    }

    [System.Threading.Tasks.Task]::WaitAll([System.Threading.Tasks.Task[]]$tasks.ToArray())
    $sw.Stop()

    $resps = @()
    foreach ($t in $tasks) {
        $body = $t.Result.Content.ReadAsStringAsync().Result
        $resps += ($body | ConvertFrom-Json)
    }
    $client.Dispose()

    return @{ wall = $sw.Elapsed.TotalSeconds; responses = $resps }
}

# Grava a resposta e devolve de onde ela veio: "content", "reasoning" ou "vazio".
# Existe por causa de um bug real do harness: com --reasoning-budget -1 o modelo gasta o
# max_tokens todo pensando, message.content sai vazio, e a carga mede geracao de tokens de
# raciocinio em vez do artefato pedido. Retornar a origem torna isso visivel no console.
function Save-Response($Resp, [string]$Path) {
    $m   = $Resp.choices[0].message
    $txt = $m.content
    $src = "content"

    if ([string]::IsNullOrWhiteSpace($txt)) {
        $temRc = $m.PSObject.Properties.Name -contains "reasoning_content"
        if ($temRc -and -not [string]::IsNullOrWhiteSpace($m.reasoning_content)) {
            $txt = $m.reasoning_content
            $src = "reasoning"
        }
        else {
            $txt = ""
            $src = "vazio"
        }
    }

    $txt | Out-File -Encoding utf8 -FilePath $Path
    return $src
}
