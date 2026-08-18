@echo off
setlocal enabledelayedexpansion

rem  TurboQuant - llama-server 200k + MTP, acessivel na rede.
rem  Uso:   start-server.bat              (porta 8080, ctx 200k, MTP n=2)
rem         start-server.bat 11434        (porta do ollama, para clientes ja configurados)
rem         start-server.bat 11434 --verbose      (args extras vao para o llama-server)
rem         set NMAX=3 ^& start-server.bat
rem         set LLAMA_API_KEY=segredo ^& start-server.bat     (exige header Authorization)
rem
rem  ATENCAO: so responde nas rotas OpenAI (/v1/chat/completions) e Anthropic (/v1/messages).
rem  As rotas nativas do ollama (/api/chat, /api/tags) nao existem aqui e dao 404.

set "BIN=E:\DEV\turboquant\llama.cpp\build\bin\Release"
set "MODEL=E:/models/Qwen3.8-27B-Q4_K_M.gguf"
set "LOGDIR=E:\DEV\turboquant\logs"

if "%PORT%"=="" set "PORT=8080"
if "%NMAX%"=="" set "NMAX=2"
if "%ALIAS%"=="" set "ALIAS=qwen3.8-27b"

rem  Tipo do KV cache. q4_0 = 4.5 bpw (padrao, seguro). q2_1 = 2.25 bpw, dobra o
rem  contexto por 4.86%% de perplexidade - ver a secao do q2_1 no README.
if "%KV%"=="" set "KV=q4_0"
rem  180k e o padrao: cabe com o VS Code fechado. Com 200k + MTP o pico bate 23.9 GB,
rem  o desktop fica sem VRAM, o WDDM pagina para a RAM e a maquina engasga.
rem  Outros contextos:  set CTX=131072 ^& start-server.bat   (folgado, da para usar a GPU junto)
rem                     set CTX=200000 ^& start-server.bat   (so com a GPU limpa)
rem  Com KV=q2_1 o padrao sobe para 262144, o teto do servidor (acima disso ele capa
rem  o slot no contexto de treino e ignora o YaRN; use llama-completion).
rem  Com visao ligada o padrao cai para 128k, porque o projetor come ~885 MiB.
if "%CTX%"=="" (
    if "%MMPROJ%"=="" (
        if "%KV%"=="q2_1" ( set "CTX=262144" ) else ( set "CTX=180000" )
    ) else ( set "CTX=131072" )
)

rem  Visao: set MMPROJ=E:/models/mmproj-F16.gguf ^& start-server.bat 11434
set "VISION="
if not "%MMPROJ%"=="" set "VISION=--mmproj %MMPROJ%"

rem  primeiro argumento numerico vira a porta; o resto e repassado ao llama-server
set "EXTRA=%*"
if not "%~1"=="" (
    echo %~1| findstr /r "^[0-9][0-9]*$" >nul
    if not errorlevel 1 (
        set "PORT=%~1"
        set "EXTRA="
        for /f "tokens=1,* delims= " %%a in ("%*") do set "EXTRA=%%b"
    )
)

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

set "AUTH="
if not "%LLAMA_API_KEY%"=="" set "AUTH=--api-key %LLAMA_API_KEY%"

for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 ^| Where-Object { $_.IPAddress -notmatch '^^(127\.|169\.254\.|172\.2[0-9]\.)' } ^| Select-Object -First 1).IPAddress"`) do set "LANIP=%%i"

echo.
echo   modelo : %ALIAS%  ^(%MODEL%^)
echo   ctx    : !CTX!   KV !KV!   MTP n=%NMAX%
echo   local  : http://127.0.0.1:%PORT%
if not "%LANIP%"=="" echo   rede   : http://%LANIP%:%PORT%
if "%AUTH%"=="" echo   AVISO  : sem API key, qualquer um na rede pode usar. Defina LLAMA_API_KEY para exigir token.
echo.
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo.

"%BIN%\llama-server.exe" ^
  -m "%MODEL%" ^
  -a "%ALIAS%" ^
  --chat-template-file "E:\DEV\turboquant\qwen35-tolerant.jinja" ^
  -c !CTX! ^
  -np 1 ^
  -ctk !KV! -ctv !KV! ^
  -ctkd !KV! -ctvd !KV! ^
  -fa on ^
  -ngl 99 ^
  -fit off ^
  -b 512 -ub 512 ^
  --spec-type draft-mtp ^
  --spec-draft-n-max %NMAX% ^
  --log-file "%LOGDIR%\server-mtp.log" ^
  --host 0.0.0.0 --port !PORT! %AUTH% !VISION! !EXTRA!

endlocal
