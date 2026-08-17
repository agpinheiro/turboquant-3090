# A/B de qualidade do KV cache. Responde: quanto a rotacao de Hadamard (estagio 1 do TurboQuant,
# ja no upstream) esta salvando, e quanta margem sobra para um quantizador melhor (estagio 2).
#
# A leitura que importa:
#   f16            = referencia sem perda
#   q4_0 + rot     = o que rodamos hoje
#   q4_0 sem rot   = o mesmo sem o estagio 1  -> a diferenca e o que a rotacao ja entrega
#   q8_0 + rot     = teto pratico de um KV quantizado
#
# A distancia entre "q4_0 + rot" e "f16" e o PREMIO MAXIMO possivel para o estagio 2.
# Se for pequena, nao ha o que ganhar escrevendo kernels novos.

param(
    [string]$File   = "E:\DEV\turboquant\data\warpeace.txt",
    [int]   $Ctx    = 32768,
    [int]   $Chunks = 8,
    [string]$Out    = "E:\DEV\turboquant\ab-perplexity.md"
)

$ErrorActionPreference = "Continue"
$bin   = "E:\DEV\turboquant\llama.cpp\build\bin\Release"
$model = "E:/models/Qwen3.8-27B-Q4_K_M.gguf"

$configs = @(
    @{ nome = "f16 (referencia)";  ctk = "f16";  ctv = "f16";  rot = $null },
    @{ nome = "q8_0 + rotacao";    ctk = "q8_0"; ctv = "q8_0"; rot = $null },
    @{ nome = "q4_0 + rotacao";    ctk = "q4_0"; ctv = "q4_0"; rot = $null },
    @{ nome = "q4_0 SEM rotacao";  ctk = "q4_0"; ctv = "q4_0"; rot = "1"  }
)

$res = @()
foreach ($c in $configs) {
    if ($c.rot) { $env:LLAMA_ATTN_ROT_DISABLE = $c.rot } else { Remove-Item Env:\LLAMA_ATTN_ROT_DISABLE -ErrorAction SilentlyContinue }

    $log = "E:\DEV\turboquant\logs\ppl-$($c.ctk)-$(if ($c.rot) {'norot'} else {'rot'}).log"
    Write-Host "`n=== $($c.nome) ===" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    & "$bin\llama-perplexity.exe" -m $model -f $File -c $Ctx --chunks $Chunks `
        -ctk $c.ctk -ctv $c.ctv -fa on -ngl 99 -fit off -b 512 -ub 512 `
        --log-file $log 2>&1 | Out-Null

    $sw.Stop()
    $txt = Get-Content $log -Raw -ErrorAction SilentlyContinue
    $ppl = if ($txt -match 'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)') { $matches[1] } else { "?" }
    $err = if ($matches) { $matches[2] } else { "?" }
    $rk  = if ($txt -match 'attn_rot_k = (\d)') { $matches[1] } else { "?" }

    $res += [pscustomobject]@{ config = $c.nome; ppl = $ppl; erro = $err; attn_rot_k = $rk; min = [math]::Round($sw.Elapsed.TotalMinutes,1) }
    Write-Host ("  PPL = {0} +/- {1}  (attn_rot_k={2}, {3} min)" -f $ppl, $err, $rk, [math]::Round($sw.Elapsed.TotalMinutes,1))
}

Remove-Item Env:\LLAMA_ATTN_ROT_DISABLE -ErrorAction SilentlyContinue

$md = @()
$md += "# A/B de perplexidade do KV cache"
$md += ""
$md += "Modelo: Qwen3.8-27B-Q4_K_M | ctx $Ctx | $Chunks chunks | corpus: $(Split-Path $File -Leaf)"
$md += ""
$md += "| config | PPL | +/- | attn_rot_k | min |"
$md += "|---|---|---|---|---|"
foreach ($r in $res) { $md += "| $($r.config) | $($r.ppl) | $($r.erro) | $($r.attn_rot_k) | $($r.min) |" }
$md -join "`n" | Out-File -FilePath $Out -Encoding utf8

Write-Host "`nresultado em $Out" -ForegroundColor Green
$res | Format-Table -AutoSize
