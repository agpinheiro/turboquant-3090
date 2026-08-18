@echo off
setlocal
rem  Aponta o Claude Code para o llama-server local APENAS nesta invocacao.
rem  Nao encosta em ~/.claude/settings.json: em qualquer outro terminal o
rem  claude continua falando com a Anthropic normalmente.
rem
rem  Uso:  claude-local                    (porta 11434)
rem        claude-local -p "resuma isso"   (args extras vao para o claude)
rem        set LLPORT=8080 ^& claude-local
rem
rem  Para ter na PATH:  copy claude-local.cmd %USERPROFILE%\.local\bin\

if "%LLPORT%"==""  set "LLPORT=11434"
if "%LLMODEL%"=="" set "LLMODEL=qwen3.8-27b"

curl -s -o nul --max-time 3 "http://127.0.0.1:%LLPORT%/props"
if errorlevel 1 (
    echo.
    echo   ERRO: nada respondendo em http://127.0.0.1:%LLPORT%
    echo   Suba o servidor primeiro:  start-server.bat %LLPORT%
    echo.
    exit /b 1
)

rem  Estas variaveis valem so para o processo filho - o setlocal/endlocal
rem  garante que nao vazam para o terminal.
set "ANTHROPIC_BASE_URL=http://127.0.0.1:%LLPORT%"
set "ANTHROPIC_AUTH_TOKEN=dummy"
set "ANTHROPIC_MODEL=%LLMODEL%"
set "ANTHROPIC_DEFAULT_HAIKU_MODEL=%LLMODEL%"

rem  Faz o modelo aparecer com nome legivel na lista do /model, em vez de
rem  "Custom model (qwen3.8-27b)".
set "ANTHROPIC_CUSTOM_MODEL_OPTION=%LLMODEL%"
set "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME=Qwen3.8-27B (local)"
set "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION=llama-server em 127.0.0.1:%LLPORT%"

claude.exe %*
endlocal
