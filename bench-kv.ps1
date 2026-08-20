# Bench do TIPO DE KV: q8_0, q4_0 e q2_1 mudam o desempenho, ou so a VRAM?
#
# Ate aqui o tipo de KV foi escolhido por MEMORIA (q2_1 cabe em 490k, q4_0 em 295k) e por
# QUALIDADE (ab-q2.ps1 / ab-perplexity.ps1: PPL +4.86% do q2_1 contra +0.16% do q4_0). O eixo
# que ninguem mediu e o TEMPO. E ele nao e obvio, porque os dois efeitos brigam:
#
#   - menos bits = menos bytes lidos por passo. As 16 camadas com KV releem o cache inteiro a
#     cada token gerado; a 70k de contexto isso e a maior parte do passo. Por ai o q2_1 ganha.
#   - menos bits = mais trabalho por byte. O kernel de flash-attention precisa dequantizar, e
#     o q2_1 usa codebook (bloco de 64 em 18 bytes) em vez do caminho direto do q8_0/q4_0.
#     Por ai o q8_0 ganha.
#
# Qual domina depende da profundidade do contexto: a contexto vazio so sobra o custo por byte,
# a contexto cheio so importa a banda. Por isso o script mede as DUAS pontas na mesma subida
# do server, e o relatorio mostra o custo MARGINAL do contexto (us por passo por 1k tokens),
# que e o numero que separa um efeito do outro.
#
# O MTP fica em n=4 FIXO (constante $MTP_N_MAX abaixo) - e o que torna a comparacao valida:
# so o tipo de KV varia. Atencao ao ler os t/s absolutos: o otimo medido em 2026-08-17 foi
# n=2 (60.1 t/s contra 53.4 do n=3), entao os numeros daqui NAO sao comparaveis com a tabela
# de baselines do CLAUDE.md. O que se compara aqui e coluna contra coluna, dentro do bench.
#
# f16 fica de fora de proposito: 6.8 GiB de KV a 98304 + modelo + compute + MTP passa dos
# 22.9 GiB livres da placa. Nao ha configuracao em que ele seja alternativa aqui.
#
# Uso:
#   .\bench-kv.ps1                            q8_0, q4_0 e q2_1 a 98304
#   .\bench-kv.ps1 -Kvs q4_0,q2_1             so os dois finalistas
#   .\bench-kv.ps1 -Ctx 65536                 mais folga de VRAM, sinal mais fraco
#   .\bench-kv.ps1 -DryRun                    estima a VRAM e imprime os comandos, nao carrega
#
# SEGURANCA: aqui a VRAM MUDA de variante para variante (esse e o ponto do bench), entao o
# script faz um pre-voo com llama-fit-params ANTES de carregar qualquer coisa e PULA o tipo
# que nao deixaria -MinFree MiB livres. Isso importa porque o modo de falha desta placa nao e
# OOM: o driver comeca a paginar VRAM para a RAM e a maquina inteira trava sem escrever erro
# nenhum no log. O ctx padrao de 98304 e o maior que deixa o q8_0 (o mais gordo dos tres)
# com ~2.3 GiB livres; a 131072 ele cai para ~1.0 GiB, que e a faixa que ja travou o PC.

param(
    [string[]] $Kvs          = @("q8_0", "q4_0", "q2_1"),
    [int]      $Ctx          = 98304,
    [int]      $PromptChars  = 300000,   # ~71k tokens em warpeace-700k.txt (~4.2 chars/token)
    [int]      $Port         = 8099,
    [string]   $Model        = "E:/models/Qwen3.8-27B-Q4_K_M.gguf",
    [int]      $MinFree      = 1500,
    [int]      $LoadTimeout  = 420,
    # "off" desliga o thinking. Sem isso o modelo gasta o max_tokens inteiro raciocinando,
    # message.content sai vazio, e o que se mede e geracao de raciocinio (ver bench-spec.ps1).
    [string]   $Reasoning    = "off",
    [string]   $OutDir       = "E:\DEV\turboquant\logs\bench-kv",
    [switch]   $DryRun
)

$ErrorActionPreference = "Stop"

# helpers compartilhados: Start-BenchServer, New-ServerArgs, Invoke-Chat, Get-FreeVram, ...
. (Join-Path $PSScriptRoot "bench-spec-common.ps1")

# ---------------------------------------------------------------- constantes do experimento
#
# MTP n=4 fixo em todas as variantes. New-ServerArgs enxerga "draft-mtp" e acrescenta sozinho
# -ctkd/-ctvd com o MESMO tipo de KV da variante - obrigatorio, senao o contexto de draft nasce
# em f16 (common/speculative.cpp) e come ~1.5 GiB a mais, o que sozinho ja invalidaria a
# comparacao de VRAM entre os tipos.

$MTP_N_MAX = 4
$SPEC_ARGS = @("--spec-type", "draft-mtp", "--spec-draft-n-max", "$MTP_N_MAX")

# O llama-fit-params nao sabe do MTP: ele estima modelo + KV + compute do alvo e mais nada.
# Medido: o MTP custa ~1.5-2 GiB fixos, nao proporcionais ao contexto. Reservamos 1900 MiB no
# pre-voo para nao aprovar uma configuracao que so vai estourar depois de carregada.
$MTP_VRAM_MIB = 1900

$FIT_BIN = "E:\DEV\turboquant\llama.cpp\build\bin\Release\llama-fit-params.exe"

# bits por peso de cada tipo, so para o relatorio (KiB/token medido vem do fit-params)
$BPW = @{ "f16" = 16.0; "q8_0" = 8.5; "q5_1" = 6.0; "q4_0" = 4.5; "q2_0" = 2.25; "q2_1" = 2.25 }

# ---------------------------------------------------------------- pre-voo de VRAM
#
# Instantaneo: o fit-params le o GGUF e faz conta, nao carrega o modelo. Saida em stdout e
# uma linha por dispositivo: "CUDA0 <modelo> <contexto> <compute>" em MiB.

function Get-FitEstimate([string]$Mdl, [int]$C, [string]$Kv) {
    # O fit-params manda a estimativa para stdout e o log para stderr. No PS 5.1 qualquer
    # redirecionamento de stderr de um .exe vira ErrorRecord, e com $ErrorActionPreference =
    # "Stop" (o padrao deste script) isso aborta tudo na primeira linha de log. A preferencia
    # abaixo e local a funcao, entao vale so para esta chamada.
    $ErrorActionPreference = "Continue"
    $out = & $FIT_BIN -m $Mdl -c $C -ctk $Kv -ctv $Kv -fa on --fit-print on 2>&1
    $m = $out | Select-String -Pattern '^\s*CUDA0\s+(\d+)\s+(\d+)\s+(\d+)' | Select-Object -First 1
    if ($null -eq $m) { return $null }
    $g = $m.Matches[0].Groups
    $modelo  = [int]$g[1].Value
    $kvMib   = [int]$g[2].Value
    $compute = [int]$g[3].Value
    return [pscustomobject]@{
        modelo    = $modelo
        kv_mib    = $kvMib
        compute   = $compute
        alvo      = $modelo + $kvMib + $compute
        com_mtp   = $modelo + $kvMib + $compute + $MTP_VRAM_MIB
        kib_token = [math]::Round($kvMib * 1024.0 / $C, 2)
    }
}

# ---------------------------------------------------------------- turnos
#
# Os quatro turnos rodam na MESMA subida do server e compartilham o prefixo gigante, entao o
# prefill de ~2 min e pago uma vez por tipo de KV (cache_prompt). A ordem importa: 'curto' vem
# ANTES do prefixo entrar no cache, senao ele mediria geracao com o contexto ja cheio.

function Get-Turnos([string]$Prefixo, [string]$Curto) {
    $pedidoNovo = "`n`n-----`nEscreva em portugues uma analise original de 200 palavras sobre a estrutura social retratada no texto acima. Nao cite trechos, nao copie frases."

    @(
        # Descartado. A primeira requisicao depois de carregar constroi os grafos CUDA
        # ("graphs reused" comeca em zero). Medido: sem este turno o 'curto' saiu a 38.66 t/s,
        # com ele a 41.99 - 8% de diferenca, mais que o efeito que o bench quer medir, e com o
        # sinal ERRADO, porque o custo do warmup cai justamente no lado raso da conta.
        @{ id = "warmup"; warmup = $true; medir = $false; max_tok = 8; prompt = "Responda apenas: ok." }

        # PONTA RASA: contexto quase zero. O KV lido por passo e desprezivel, entao o que
        # sobra e o custo fixo do tipo (dequantizacao por byte). Se q2_1 perder AQUI, o
        # kernel de codebook e mais caro - e a perda some quando o contexto cresce.
        @{ id = "curto";  medir = $true;  max_tok = 300; prompt = $Curto + $pedidoNovo }

        # nao medido para geracao, mas o prompt_per_second daqui e um resultado: o prefill
        # tambem passa pelo tipo de KV (escreve o cache, e dequantiza uma camada por vez)
        @{ id = "prefill"; medir = $false; max_tok = 8;   prompt = $Prefixo + "`n`n-----`nResponda apenas: ok." }

        # PONTA CHEIA, mesmo pedido do 'curto': a unica diferenca e a profundidade do contexto.
        # E esse par (curto, cheio) que da o custo marginal do KV por 1k tokens.
        @{ id = "cheio";  medir = $true;  max_tok = 250; prompt = $Prefixo + $pedidoNovo }

        # saida quase toda copiada do contexto: aceitacao de draft alta. Serve de controle -
        # se a aceitacao mudar muito entre os tipos de KV, o t/s deixa de medir so a banda.
        @{
            id = "citar"; medir = $true; max_tok = 400
            prompt = $Prefixo + "`n`n-----`nExtraia do texto acima 6 trechos literais longos (copiados exatamente, entre aspas, pelo menos 25 palavras cada) que envolvam Napoleao ou a guerra. So os trechos, um por linha."
        }
    )
}

function New-Row($Kv, $Turno, $Status, $Free) {
    return [pscustomobject]@{
        kv = $Kv; turno = $Turno; status = $Status
        gen_ts = $null; prompt_ts = $null; prompt_n = $null; cache_n = $null
        draft_acc = $null; draft_n = $null; draft_ok = $null; gen_tok = $null
        kv_mib = $null; kib_token = $null; free_mib = $Free
    }
}

# ---------------------------------------------------------------- preparo

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

$fonte = Join-Path $script:BENCH_DATA "warpeace-700k.txt"
$texto = Get-Content -Raw -Path $fonte
if ($texto.Length -lt $PromptChars) {
    Write-Error "$fonte tem so $($texto.Length) chars, menos que -PromptChars $PromptChars"
    return
}
$prefixo = $texto.Substring(0, $PromptChars)
$curto   = $texto.Substring(0, 2000)
$turnos  = Get-Turnos $prefixo $curto

Write-Host ""
Write-Host "  modelo   : $Model" -ForegroundColor Cyan
Write-Host "  ctx      : $Ctx   porta $Port   MTP n=$MTP_N_MAX (constante)" -ForegroundColor Cyan
Write-Host "  tipos KV : $($Kvs -join ', ')" -ForegroundColor Cyan
Write-Host "  prefixo  : $PromptChars chars (~$([math]::Round($PromptChars/4.2/1000))k tokens) de $(Split-Path $fonte -Leaf)" -ForegroundColor Cyan
Write-Host "  saida    : $OutDir" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------- pre-voo
#
# Nada foi carregado ainda. Mata server orfao antes de ler a VRAM livre, senao a baseline vem
# contaminada e o pre-voo aprova o que nao cabe.

Stop-AllServers
$livre = Get-FreeVram

Write-Host "=== pre-voo (llama-fit-params, nao carrega o modelo) ===" -ForegroundColor Green
Write-Host ("  VRAM livre agora: {0} MiB   |   reserva do MTP: {1} MiB   |   margem minima: {2} MiB" -f $livre, $MTP_VRAM_MIB, $MinFree)
Write-Host ("{0,-8} {1,8} {2,8} {3,8} {4,10} {5,10} {6,10}  {7}" -f `
    "kv", "modelo", "KV", "compute", "KiB/token", "com MTP", "sobra", "veredito")

$plano = @()
foreach ($kv in $Kvs) {
    $est = Get-FitEstimate -Mdl $Model -C $Ctx -Kv $kv
    if ($null -eq $est) {
        Write-Host ("{0,-8} {1}" -f $kv, "fit-params nao respondeu (tipo invalido nesta build?)") -ForegroundColor Red
        continue
    }
    $sobra = $livre - $est.com_mtp
    $cabe  = $sobra -ge $MinFree
    $cor   = if ($cabe) { "Gray" } else { "Red" }
    $ver   = if ($cabe) { "ok" } else { "PULA - reduza -Ctx" }
    Write-Host ("{0,-8} {1,8} {2,8} {3,8} {4,10:N2} {5,10} {6,10} {7,10}" -f `
        $kv, $est.modelo, $est.kv_mib, $est.compute, $est.kib_token, $est.com_mtp, $sobra, $ver) -ForegroundColor $cor
    if ($cabe) { $plano += @{ kv = $kv; est = $est } }
}
Write-Host ""

if ($plano.Count -eq 0) {
    Write-Error "nenhum tipo de KV cabe com margem de $MinFree MiB. Reduza -Ctx ou feche o VS Code."
    return
}

# ---------------------------------------------------------------- execucao

$results = @()

foreach ($p in $plano) {
    $kv      = $p.kv
    $log     = Join-Path $OutDir ("server-" + $kv + ".log")
    $allArgs = New-ServerArgs -SpecArgs $SPEC_ARGS -Model $Model -Ctx $Ctx -Kv $kv -Port $Port `
                              -LogPath $log -Reasoning $Reasoning

    Write-Host ("=== KV {0}  ({1} bpw, {2} MiB de cache estimados)" -f `
        $kv, $BPW[$kv], $p.est.kv_mib) -ForegroundColor Yellow

    if ($DryRun) {
        Write-Host ("    " + $script:BENCH_BIN + " " + ($allArgs -join " ")) -ForegroundColor DarkGray
        continue
    }

    Remove-Item $log -ErrorAction SilentlyContinue
    $up = Start-BenchServer -AllArgs $allArgs -Port $Port -TimeoutSec $LoadTimeout -MinFree $MinFree

    if ($up.status -ne "ok") {
        if ($up.status -eq "vram-unsafe") {
            Write-Host ("    ABORTADA: so {0} MiB livres (minimo {1}), o pre-voo subestimou. Reduza -Ctx." -f `
                $up.free, $MinFree) -ForegroundColor Red
        }
        else {
            Write-Host ("    FALHOU ao subir ({0}) - ver {1}" -f $up.status, $log) -ForegroundColor Red
        }
        $results += New-Row $kv "-" $up.status $up.free
        Stop-AllServers
        continue
    }

    Write-Host ("    carregou, {0} MiB livres (pre-voo previa {1})" -f `
        $up.free, ($livre - $p.est.com_mtp)) -ForegroundColor DarkGray

    foreach ($tn in $turnos) {
        try {
            $r = Invoke-Chat -Port $Port -Prompt $tn.prompt -MaxTok $tn.max_tok
            $t = $r.timings

            if ($tn.warmup) {
                Write-Host ("    {0,-8} descartado, so constroi os grafos ({1:N2} t/s)" -f `
                    $tn.id, $t.predicted_per_second) -ForegroundColor DarkGray
                continue
            }

            $acc = $null
            if ($t.draft_n -gt 0) { $acc = [math]::Round($t.draft_n_accepted / $t.draft_n, 4) }

            $row = [pscustomobject]@{
                kv        = $kv
                turno     = $tn.id
                status    = "ok"
                gen_ts    = if ($tn.medir) { [math]::Round($t.predicted_per_second, 2) } else { $null }
                prompt_ts = [math]::Round($t.prompt_per_second, 1)
                prompt_n  = $t.prompt_n
                cache_n   = $t.cache_n
                draft_acc = $acc
                draft_n   = $t.draft_n
                draft_ok  = $t.draft_n_accepted
                gen_tok   = $t.predicted_n
                kv_mib    = $p.est.kv_mib
                kib_token = $p.est.kib_token
                free_mib  = $up.free
            }
            $results += $row

            if ($tn.medir) {
                $origem = Save-Response $r (Join-Path $OutDir ("resp-" + $kv + "-" + $tn.id + ".txt"))
                $aviso  = ""
                if ($origem -ne "content") { $aviso = "   <-- saida em '$origem', nao em content!" }
                Write-Host ("    {0,-8} {1,7:N2} t/s   draft {2}/{3}   ctx {4:N0}{5}" -f `
                    $tn.id, $row.gen_ts, $t.draft_n_accepted, $t.draft_n, ($t.prompt_n + $t.cache_n), $aviso)
            }
            else {
                Write-Host ("    {0,-8} {1,7:N0} tokens de prompt a {2,6:N0} t/s" -f `
                    $tn.id, $t.prompt_n, $t.prompt_per_second) -ForegroundColor DarkGray
            }
        }
        catch {
            Write-Host ("    {0,-8} ERRO: {1}" -f $tn.id, $_.Exception.Message) -ForegroundColor Red
            $results += New-Row $kv $tn.id "erro" $up.free
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

$feitos = @($ok | ForEach-Object { $_.kv } | Select-Object -Unique)

function Get-Cell($Kv, $Turno) {
    return $ok | Where-Object { $_.kv -eq $Kv -and $_.turno -eq $Turno } | Select-Object -First 1
}

# rotulo com a profundidade REAL medida, nao a estimada por -PromptChars
function Get-Rotulo([string]$Turno, [string]$Nome) {
    $r = $ok | Where-Object { $_.turno -eq $Turno } | Select-Object -First 1
    if ($null -eq $r) { return $Nome }
    $n = $r.prompt_n + $r.cache_n
    $s = if ($n -ge 2000) { "{0:N0}k" -f ($n / 1000) } else { "{0:N0}" -f $n }
    return "$Nome ($s tok)"
}

# baseline = q4_0 se estiver no lote (e a config em producao hoje), senao o primeiro medido
$baseKv = if ($feitos -contains "q4_0") { "q4_0" } else { $feitos[0] }

$fmt = "{0,-8} {1,20} {2,20} {3,20} {4,16}"

Write-Host "=== t/s de geracao (ganho vs $baseKv) ===" -ForegroundColor Green
Write-Host ($fmt -f "kv", (Get-Rotulo "curto" "curto"), (Get-Rotulo "cheio" "cheio"), `
                    (Get-Rotulo "citar" "citar"), "prefill t/s")
foreach ($kv in $feitos) {
    $cells = @()
    foreach ($tid in @("curto", "cheio", "citar")) {
        $r = Get-Cell $kv $tid
        if ($null -eq $r -or $null -eq $r.gen_ts) { $cells += "-"; continue }
        $cell = "{0,7:N2}" -f $r.gen_ts
        $b = Get-Cell $baseKv $tid
        if ($kv -ne $baseKv -and $null -ne $b -and $b.gen_ts -gt 0) {
            $cell += "  ({0,6:P0})" -f (($r.gen_ts / $b.gen_ts) - 1)
        }
        $cells += $cell
    }
    $pf = Get-Cell $kv "prefill"
    $cells += if ($null -eq $pf) { "-" } else { "{0,10:N0}" -f $pf.prompt_ts }
    Write-Host ($fmt -f $kv, $cells[0], $cells[1], $cells[2], $cells[3])
}

# ---- por token gerado: o que o usuario sente, mas NAO responde a pergunta ----
#
# 'curto' e 'cheio' pedem a MESMA coisa; so a profundidade do contexto muda. A tentacao e ler
# a diferenca de ms por token gerado como "o custo de reler o KV". Nao e: com MTP no meio, o
# t/s tambem depende de quantos tokens cada passo entrega, e isso e a aceitacao de draft - que
# muda com o tipo de KV, porque a cabeca MTP le o mesmo cache quantizado.
#
# Medido em 2026-08-19: por esta metrica o q2_1 aparece com 82 us/1k contra 167 do q4_0, o que
# se leria como "q2_1 e 2x mais barato". A tabela por passo, logo abaixo, mostra o contrario.
# Esta fica so como o numero que o usuario de fato sente.

function Get-Passos($R) {
    # p_min = 0 em todas as variantes deste bench, entao o drafter sempre rascunha n_max
    # tokens e draft_n / n_max e o numero exato de passos do modelo alvo.
    if ($null -eq $R.draft_n -or $R.draft_n -le 0) { return [double]$R.gen_tok }  # sem spec
    return [double]$R.draft_n / $MTP_N_MAX
}

Write-Host ""
Write-Host "=== custo marginal do contexto, por TOKEN GERADO (confundido com a aceitacao) ===" -ForegroundColor Green
Write-Host ("{0,-8} {1,12} {2,12} {3,14} {4,18} {5,12}" -f `
    "kv", "ms/tok raso", "ms/tok fundo", "delta ms/tok", "us por 1k de ctx", "KiB/token")
$temNegativo = $false
foreach ($kv in $feitos) {
    $c = Get-Cell $kv "curto"
    $f = Get-Cell $kv "cheio"
    $p = Get-Cell $kv "prefill"
    if ($null -eq $c -or $null -eq $f -or $c.gen_ts -le 0 -or $f.gen_ts -le 0) {
        Write-Host ("{0,-8} {1}" -f $kv, "faltou medida"); continue
    }
    $ctxTok = if ($null -ne $p) { $p.prompt_n + $p.cache_n } else { [int]($PromptChars / 4.2) }
    $msRaso  = 1000.0 / $c.gen_ts
    $msFundo = 1000.0 / $f.gen_ts
    $delta   = $msFundo - $msRaso
    $por1k   = $delta * 1000.0 / ($ctxTok / 1000.0)
    if ($delta -le 0) { $temNegativo = $true }
    Write-Host ("{0,-8} {1,12:N2} {2,12:N2} {3,14:N2} {4,18:N1} {5,12:N2}" -f `
        $kv, $msRaso, $msFundo, $delta, $por1k, $c.kib_token)
}
if ($temNegativo) {
    Write-Host ("  delta <= 0: o prefixo esta raso demais para o efeito sair do ruido " +
                "(a ~10k tokens ele vale ~0.1 ms/token). Aumente -PromptChars e -Ctx.") -ForegroundColor Yellow
}

# ---- por PASSO do modelo: a resposta ----
#
# Dividir o tempo pelo numero de passos (draft_n / n_max) tira a aceitacao da conta e sobra so
# o custo de um forward pass. Se o tipo de KV importasse por banda, o passo fundo teria que
# cair com os bits: o q8_0 le 35.6 KiB/token de contexto, o q2_1 le 10.6.

Write-Host ""
Write-Host "=== custo por PASSO do modelo (sem o efeito da aceitacao) ===" -ForegroundColor Green
Write-Host ("{0,-8} {1,10} {2,10} {3,12} {4,12} {5,12} {6,18}" -f `
    "kv", "tok/passo", "  (fundo)", "ms/passo", "  (fundo)", "delta ms", "us por 1k de ctx")
foreach ($kv in $feitos) {
    $c = Get-Cell $kv "curto"
    $f = Get-Cell $kv "cheio"
    $p = Get-Cell $kv "prefill"
    if ($null -eq $c -or $null -eq $f -or $c.gen_ts -le 0 -or $f.gen_ts -le 0) {
        Write-Host ("{0,-8} {1}" -f $kv, "faltou medida"); continue
    }
    $ctxTok    = if ($null -ne $p) { $p.prompt_n + $p.cache_n } else { [int]($PromptChars / 4.2) }
    $passosR   = Get-Passos $c
    $passosF   = Get-Passos $f
    $tokPassoR = $c.gen_tok / $passosR
    $tokPassoF = $f.gen_tok / $passosF
    $msPassoR  = 1000.0 * ($c.gen_tok / $c.gen_ts) / $passosR
    $msPassoF  = 1000.0 * ($f.gen_tok / $f.gen_ts) / $passosF
    $delta     = $msPassoF - $msPassoR
    $por1k     = $delta * 1000.0 / ($ctxTok / 1000.0)
    Write-Host ("{0,-8} {1,10:N2} {2,10:N2} {3,12:N2} {4,12:N2} {5,12:N2} {6,18:N1}" -f `
        $kv, $tokPassoR, $tokPassoF, $msPassoR, $msPassoF, $delta, $por1k)
}

Write-Host ""
Write-Host "=== aceitacao de draft (MTP n=$MTP_N_MAX em todos) ===" -ForegroundColor Green
Write-Host ($fmt -f "kv", (Get-Rotulo "curto" "curto"), (Get-Rotulo "cheio" "cheio"), `
                    (Get-Rotulo "citar" "citar"), "")
foreach ($kv in $feitos) {
    $cells = @()
    foreach ($tid in @("curto", "cheio", "citar")) {
        $r = Get-Cell $kv $tid
        if ($null -eq $r -or $null -eq $r.draft_acc) { $cells += "-"; continue }
        $cells += "{0,6:P1}  {1}/{2}" -f $r.draft_acc, $r.draft_ok, $r.draft_n
    }
    Write-Host ($fmt -f $kv, $cells[0], $cells[1], $cells[2], "")
}

Write-Host ""
Write-Host "=== VRAM ===" -ForegroundColor Green
Write-Host ("{0,-8} {1,12} {2,12} {3,14}" -f "kv", "KV MiB", "KiB/token", "livre MiB")
foreach ($kv in $feitos) {
    $r = Get-Cell $kv "cheio"
    if ($null -eq $r) { $r = $ok | Where-Object { $_.kv -eq $kv } | Select-Object -First 1 }
    Write-Host ("{0,-8} {1,12} {2,12:N2} {3,14}" -f $kv, $r.kv_mib, $r.kib_token, $r.free_mib)
}

Write-Host ""
Write-Host "Leitura (a ordem importa - a tabela por passo manda):" -ForegroundColor DarkGray
Write-Host "  1. ms/passo igual entre os tipos = o tipo de KV NAO muda o custo do forward pass, e" -ForegroundColor DarkGray
Write-Host "     toda diferenca de t/s que aparecer na primeira tabela e aceitacao de draft." -ForegroundColor DarkGray
Write-Host "  2. ms/passo fundo caindo com os bits = ai sim o kernel e limitado por banda, e o KV" -ForegroundColor DarkGray
Write-Host "     mais estreito paga em velocidade alem de pagar em VRAM." -ForegroundColor DarkGray
Write-Host "  3. tok/passo caindo com os bits (medido em 2026-08-19) = o KV grosseiro degrada a" -ForegroundColor DarkGray
Write-Host "     cabeca MTP. Isso e QUALIDADE aparecendo como velocidade, e o remedio nao e trocar" -ForegroundColor DarkGray
Write-Host "     o KV: e baixar o -n-max, que com aceitacao ruim desperdica menos verificacao." -ForegroundColor DarkGray
Write-Host "  Qualidade direta nao e medida aqui: para isso ab-q2.ps1 (PPL) e ab-perplexity.ps1." -ForegroundColor DarkGray
