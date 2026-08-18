# Portao de decisao da Fase 1: qualidade do KV em q2_0 (2.25 bpw), medida no modelo real
# sem escrever um kernel CUDA.
#
# O q2_0 ja existe como tipo ggml (bloco de 64 + escala fp16) com caminho de CPU completo.
# Falta so o flash-attention em CUDA. Entao rodamos o KV na CPU via -nkvo, que e lento
# mas responde a unica pergunta que importa antes de investir nos kernels.
#
# Baseline e q4_0 TAMBEM com -nkvo, para isolar a quantizacao do caminho de codigo.

param(
    [string]$Model  = "E:/models/Qwen3.8-27B-Q4_K_M.gguf",
    [string]$Corpus = "E:\DEV\turboquant\data\warpeace.txt",
    [int]   $Ctx    = 8192,
    [int]   $Chunks = 4,
    [string]$Out    = "E:\DEV\turboquant\ab-q2.md"
)

$ErrorActionPreference = "Continue"
$bin = "E:\DEV\turboquant\llama.cpp\build\bin\Release"

$res = @()
foreach ($t in 'f16', 'q4_0', 'q2_0') {
    $log = "E:\DEV\turboquant\logs\ppl-nkvo-$t.log"
    Write-Host "`n=== KV $t (na CPU) ===" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    & "$bin\llama-perplexity.exe" -m $Model -f $Corpus -c $Ctx --chunks $Chunks `
        -ctk $t -ctv $t -nkvo -ngl 99 -fit off -b 512 -ub 512 `
        --log-file $log 2>&1 | Out-Null

    $sw.Stop()
    $txt = Get-Content $log -Raw -ErrorAction SilentlyContinue
    if ($txt -match 'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)') {
        $ppl, $err = $matches[1], $matches[2]
    } else {
        $ppl, $err = 'FALHOU', '-'
        Write-Host "  sem resultado - ultimas linhas do log:" -ForegroundColor Yellow
        Get-Content $log -Tail 4 -ErrorAction SilentlyContinue | ForEach-Object { "    $_" }
    }
    $rot = if ($txt -match 'attn_rot_k = (\d)') { $matches[1] } else { '?' }
    $res += [pscustomobject]@{ kv = $t; ppl = $ppl; erro = $err; rot = $rot; min = [math]::Round($sw.Elapsed.TotalMinutes, 1) }
    Write-Host ("  PPL = {0} +/- {1}  (attn_rot_k={2}, {3} min)" -f $ppl, $err, $rot, [math]::Round($sw.Elapsed.TotalMinutes, 1))
}

$base = ($res | Where-Object { $_.kv -eq 'f16' }).ppl
$md = @("# KV q2_0 - portao de decisao da Fase 1", "",
        "Modelo: $(Split-Path $Model -Leaf) | ctx $Ctx | $Chunks chunks | KV na CPU (-nkvo)", "",
        "| KV | bpw | PPL | +/- | delta vs f16 | min |", "|---|---|---|---|---|---|")
foreach ($r in $res) {
    $bpw = switch ($r.kv) { 'f16' { '16' } 'q4_0' { '4.5' } 'q2_0' { '2.25' } }
    $d = if ($r.ppl -ne 'FALHOU' -and $base -ne 'FALHOU') {
        "{0:P3}" -f (([double]$r.ppl - [double]$base) / [double]$base)
    } else { '-' }
    $md += "| $($r.kv) | $bpw | $($r.ppl) | $($r.erro) | $d | $($r.min) |"
}
$md -join "`n" | Out-File -FilePath $Out -Encoding utf8
Write-Host "`nresultado em $Out" -ForegroundColor Green
$res | Format-Table -AutoSize
