# Recria o corpus de teste de contexto longo.
#
# Fonte: Project Gutenberg #2600 - "War and Peace", Leo Tolstoy (dominio publico).
# O arquivo inteiro tem ~3.3 MB = ~900k tokens, o que estoura qualquer contexto aqui.
# O recorte de 700 KB da 166194 tokens neste tokenizer - enche 180k/200k deixando
# folga para pergunta e resposta.

param(
    [string]$Url   = "https://www.gutenberg.org/files/2600/2600-0.txt",
    [string]$Full  = "$PSScriptRoot\warpeace.txt",
    [string]$Slice = "$PSScriptRoot\warpeace-700k.txt",
    [int]   $Bytes = 700000
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Full)) {
    Write-Host "baixando $Url ..."
    Invoke-WebRequest -Uri $Url -OutFile $Full
}

$fs  = [System.IO.File]::OpenRead($Full)
$buf = New-Object byte[] $Bytes
$n   = $fs.Read($buf, 0, $Bytes)
$fs.Close()
[System.IO.File]::WriteAllBytes($Slice, $buf[0..($n-1)])

Write-Host ("{0}: {1:N0} bytes" -f (Split-Path $Slice -Leaf), $n)
Write-Host "conferir a contagem de tokens:"
Write-Host "  llama-tokenize.exe -m <modelo.gguf> -f $Slice --ids | (contar virgulas)"
