# TurboQuant — 200k de contexto num único RTX 3090

Objetivo: rodar o **Qwen3.8-27B Q4_K_M com 200k+ tokens de contexto numa única 3090 (24 GiB)**,
sem spill para CPU, mantendo qualidade aceitável. O caminho é KV cache em 4–5 bits com rotação
de Hadamard (TurboQuant); o resto do orçamento de VRAM é praticamente fixo.

## Máquina

| | |
|---|---|
| GPU | RTX 3090, 24576 MiB (~1600 MiB já ocupados pelo desktop do Windows → **~22.9 GiB livres**) |
| RAM | 31.9 GiB + pagefile 34 GiB em `E:` |
| SO / toolchain | Windows 11 Pro, MSVC x64 Release, CUDA 13.2, `CMAKE_CUDA_ARCHITECTURES=86` |
| llama.cpp | `llama.cpp/` (fork upstream), build `d8df12e`, `GGML_CUDA=ON`, `GGML_CUDA_FA=ON` |
| Binários | `llama.cpp/build/bin/Release/` |
| Modelo | `E:/models/Qwen3.8-27B-Q4_K_M.gguf` (15.93 GiB) |
| Dataset | `data/warpeace.txt` (3.36 MB) — para perplexidade / prompts longos |

## O modelo (lido do GGUF, não do papo)

- `general.architecture = qwen35`, `block_count = 65` (64 camadas + 1 de MTP/`nextn_predict_layers=1`)
- `full_attention_interval = 4` → **só 16 camadas têm KV cache**; as outras 48 são SSM
  (`conv_kernel=4`, `state_size=128`, `group_count=16`, `inner_size=6144`) com estado **constante**,
  que não cresce com o contexto. É isso que torna 200k viável em 24 GiB.
- `head_count=24`, `head_count_kv=4`, `key_length = value_length = 256`
- `context_length = 262144` nativo, `rope.freq_base = 1e7`, mrope `[11,11,10,0]`

KV por token = 16 camadas × 4 heads_kv × (256+256) = **32768 elementos**:

| tipo | por token | 200k tokens |
|---|---|---|
| f16 | 64 KiB | 12.4 GiB |
| q8_0 | ~34 KiB | 6.6 GiB |
| q5_1 | ~22 KiB | 4.7 GiB |
| q4_0 | ~18 KiB | 3.6 GiB |

## Orçamento de VRAM medido

Números reais de `llama-fit-params --fit-print on` (MiB em CUDA0, `-fa on`, modelo Q4_K_M):

| KV (k/v) | ctx | modelo | KV | compute | **total** | cabe? |
|---|---|---|---|---|---|---|
| f16 / f16 | 200k | 15345 | 12661 | 505 | **28511** | não |
| q8_0 / q8_0 | 200k | 15345 | 6796 | 1057 | **23198** | **não** (estoura com o desktop) |
| q8_0 / q4_0 | 200k | 15345 | 5232 | 505 | **21082** | sim |
| q5_1 / q5_1 | 200k | 15345 | 4841 | 505 | **20691** | sim |
| q4_0 / q4_0 | 200k | 15345 | 3668 | 1057 | **20070** | sim (~2.8 GiB de folga) |
| q4_0 / q4_0 | 262k | 15345 | 4757 | 1360 | **21462** | sim — contexto nativo inteiro |

Conclusão: **q8_0 não fecha em 200k**; 4–5 bits fecha, inclusive nos 262144 nativos.
Por isso a qualidade do KV em 4 bits é *o* problema do projeto, não a memória.

## Rotação de Hadamard já está no upstream

`src/llama-kv-cache.cpp` gera as matrizes de Walsh-Hadamard e liga a rotação **automaticamente**
quando o KV é quantizado e `head_dim % 64 == 0` (aqui é 256, então liga sozinho). Confirme no log:

```
llama_kv_cache: attn_rot_k = 1, ...
llama_kv_cache: attn_rot_v = 1, ...
```

- `LLAMA_ATTN_ROT_DISABLE=1` desliga → é o **A/B para medir o ganho da rotação**.
- `LLAMA_KV_CACHE_DEBUG=1` para debug do cache.
- No k-shift a rotação é desfeita e refeita em volta do RoPE (`llama_mul_mat_hadamard`) — mexer em
  context shift / RoPE exige cuidar desse par.

Não reimplementar o estágio 1 do TurboQuant. O que falta é o que vem depois (calibração/escala por
canal, 3 bits, outliers).

## Regras de execução — leia antes de rodar qualquer teste

Em 2026-08-17 um teste **travou a máquina inteira** (Kernel-Power 41 + EventLog 6008, desligamento
não esperado às 16:44, reboot 17:06). Duas causas somadas:

1. **`llama.exe` (o app interativo) foi rodado sem TTY.** Ao pegar EOF no stdin ele entra em loop
   imprimindo o prompt: `run_200k_q4.log` ficou com **618519 linhas** de `> ` (2.4 MB) e limpou o
   scrollback do terminal.
2. **A config de 200k não cabia na GPU.** Com `--fit on` (padrão) o llama.cpp não falha: ele reduz
   ctx e/ou empurra camadas para a CPU (`fit_params_target` = 1 GiB de margem, `fit_params_min_ctx`
   = 4096). O resultado foi 7.0 t/s de geração — contra **44 t/s no ollama** — e pressão de RAM/
   pagefile num host de 32 GiB até congelar.

Portanto:

- **Nunca** invocar `llama.exe` de script/agente. Use `llama-completion.exe`, `llama-cli.exe -no-cnv`,
  `llama-server.exe` ou `llama-bench.exe`.
- **Sempre** estimar antes de carregar (não carrega o modelo, é instantâneo):
  ```powershell
  .\llama-fit-params.exe -m E:/models/Qwen3.8-27B-Q4_K_M.gguf -c 200000 -ctk q4_0 -ctv q4_0 -fa on --fit-print on
  ```
  Saída = `dispositivo modelo contexto compute` em MiB. Some e compare com ~22900 MiB.
- **Sempre** fixar `-ngl 99 -fit off` nos testes de verdade. Assim uma config que não cabe morre com
  CUDA OOM em segundos, em vez de escorrer para a RAM e derrubar o Windows. Se `-ngl` for definido
  pelo usuário o fitter aborta o ajuste (`common/fit.cpp:377`) — é o comportamento que queremos.
- Limitar `-n` e redirecionar stdout+stderr para arquivo; conferir o tamanho do log depois.
- Não deixar nada além do desktop na GPU durante a medição (`nvidia-smi` antes: baseline ~1.6 GiB).

## Comandos base

Geração não-interativa, 200k, KV 4 bits (config que cabe):

```powershell
cd E:\DEV\turboquant\llama.cpp\build\bin\Release
.\llama-completion.exe -m E:/models/Qwen3.8-27B-Q4_K_M.gguf `
  -c 200000 -ctk q4_0 -ctv q4_0 -fa on -ngl 99 -fit off `
  -b 512 -ub 512 -n 128 --no-jinja --chat-template chatml `
  -f E:\DEV\turboquant\data\warpeace.txt *> E:\DEV\turboquant\logs\run.log
```

Throughput comparável ao ollama (MTP speculative decoding — é daí que vêm os 44 t/s):

```powershell
.\llama-bench.exe -m E:/models/Qwen3.8-27B-Q4_K_M.gguf `
  -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 4096 -n 128
```
Flags de MTP no CLI/server: `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling`.

A/B de qualidade do KV (o experimento central do projeto):

```powershell
# com rotação (padrão)
.\llama-perplexity.exe -m E:/models/Qwen3.8-27B-Q4_K_M.gguf -f E:\DEV\turboquant\data\warpeace.txt `
  -c 32768 -ctk q4_0 -ctv q4_0 -fa on -ngl 99 -fit off
# sem rotação
$env:LLAMA_ATTN_ROT_DISABLE=1; <mesmo comando>; Remove-Item Env:\LLAMA_ATTN_ROT_DISABLE
```

## Baselines medidos (2026-08-17, prompt curto, temp 0, n_predict 200)

| setup | ctx | KV | gen t/s | draft aceito | VRAM |
|---|---|---|---|---|---|
| `llama-cli`, sem spec | 200k | q4_0 | 35.5 | — | 20070 MiB |
| `llama-server` + `draft-mtp` n=1 | 200k | q4_0 | 55.4 | 88/110 = 80% | ~23300 MiB |
| **`llama-server` + `draft-mtp` n=2** | 200k | q4_0 | **60.1** | 115/165 = 70% | ~23100 MiB |
| `llama-server` + `draft-mtp` n=3 | 200k | q4_0 | 53.4 | 123/223 = 55% | ~23575 MiB |
| ollama (baseline anterior) | 64k | q8_0 | ~44 | — | — |
| `llama.exe`, run travado | 200k pedido | ? | 7.0 | — | camadas na CPU |

- **`--spec-draft-n-max 2` é o ótimo.** A cabeça MTP prevê 1 token (`nextn_predict_layers=1`); o 3º
  token de rascunho derruba a aceitação de 70% para 55% e a verificação desperdiçada custa mais do
  que o ganho. n=1 aceita 80% mas rascunha pouco.
- O campo `speculative.n_max` por requisição **é ignorado** — vale só o flag de startup.
- Sem spec a decodificação é limitada por banda (memory controller em 85%, 96% de GPU util): 35 t/s
  é o teto físico da 3090 para 27B denso. MTP é o único caminho acima disso.
- MTP custa **~1.8 GiB** de VRAM. Sempre com `-ctkd q4_0 -ctvd q4_0` (o KV do contexto de draft
  nasce em f16, `common/speculative.cpp:2337`) e `-np 1` (o server abre 4 slots por padrão; com 4
  sobram só 317 MiB livres).

## Orçamento real com o desktop junto

Tudo é pré-alocado: o **pico durante a geração fica ~30 MiB acima do idle**. Se carregou com folga,
não engasga depois. O que mata é carregar sem folga.

| config | VRAM total | livre | veredito |
|---|---|---|---|
| 200k + MTP | **23900 MiB** de pico | ~600 MiB | **trava o PC** — WDDM pagina VRAM para a RAM |
| **180k + MTP** | 22780 MiB de pico | ~1800 MiB | ok com o VS Code fechado ← padrão do `.bat` |
| 128k + MTP | ~21300 MiB estimado | ~3200 MiB | folgado, dá para usar a GPU junto |

O modo de falha aqui **não é OOM**: o CUDA aloca, o driver começa a paginar VRAM para a RAM e a
máquina inteira engasga (mouse travando, tela piscando). Não aparece erro nenhum no log do llama.
Por isso o alvo é deixar ≥1.5 GiB livres, não zero.

Qualquer número abaixo de ~30 t/s com tudo na GPU é sinal de que algo escorreu para a CPU — confira
`-ngl` efetivo e os buffers no log antes de investigar outra coisa.

## Contexto cheio: 166312 tokens, medido (2026-08-17)

`test-fullctx.ps1` via `/v1/chat/completions`, MTP n=2, KV q4_0:

| | |
|---|---|
| prefill | 166312 tokens em 255.8 s = **650 t/s** de média |
| geração | 331 tokens a **31.5 t/s** (contra 60.1 com contexto vazio) |
| draft aceito | 197/266 = **74%** — a aceitação **não** cai com contexto cheio |
| recall | **acertou** a pergunta sobre as primeiras páginas |

O teste de recall pergunta onde começa a narrativa; o modelo respondeu "São Petersburgo, soirée de
Anna Pávlovna Schérer" — correto, e a informação estava ~166k tokens atrás. **KV em q4_0 com rotação
de Hadamard preserva atenção de longa distância.**

Decaimento do prefill (t/s instantâneo por posição no contexto):

| pos | 16k | 33k | 49k | 66k | 82k | 98k | 115k | 131k | 147k | 164k |
|---|---|---|---|---|---|---|---|---|---|---|
| t/s | 1113 | 966 | 853 | 753 | 674 | 610 | 557 | 502 | 461 | 427 |

Ajusta bem a `ms/token = 0.74 + 0.0098 * (pos/1000)`: custo constante de pesos/FFN/SSM mais um termo
**linear** na posição, que é a atenção das 16 camadas. 10x mais contexto custa só 2.6x — num denso
seria ~10x. Extrapolando: ~372 t/s aos 200k, ~300 t/s aos 262k; encher 200k leva ~5.7 min.
Use `cache_prompt` para não pagar isso duas vezes.

## KV em 2 bits: `q2_1` (branch `kv-q2_0` do llama.cpp)

O `q4_0` é o piso do llama.cpp e limita esta placa a ~295k. Construímos um tipo de 2,25 bpw.

**Não use `q2_0` para KV.** Ele existe e os kernels CUDA que escrevemos funcionam, mas o formato
usa `d = amax` com mapeamento `(q-1)·d`, o que torna o código `11` inalcançável: uso medido de
10,2 / 79,6 / 10,2 / **0,0** %. Zera 80 % dos valores, SNR 2,88 dB, perplexidade +24,5 %.

**Use `q2_1`.** Mesmo bloco (64 valores em 18 bytes), codebook `{-10, -3, +3, +10}` com escala
`0,1510 × rms`. Inteiros porque o `vec_dot` usa `dp4a`; a razão 10/3 aproxima Lloyd-Max com 0,1 %
de erro. Uso dos códigos 16,5 / 33,5 / 33,5 / 16,5 %, SNR 9,41 dB, perplexidade **+4,86 %**.

| KV | bpw | PPL (ctx 8k, 4 chunks) | teto de contexto |
|---|---|---|---|
| f16 | 16 | 6.8594 | ~103k |
| q4_0 | 4,5 | 6.8701 | ~295k |
| q2_0 | 2,25 | 8.5422 | — (inutilizável) |
| **q2_1** | **2,25** | **7.1927** | **~490k** |

Verificado em 427.759 tokens com YaRN fator 4: prefill 381,6 t/s, geração 9,5 t/s, **recall
correto** (São Petersburgo / soirée / Anna Pávlovna Schérer, a ~428k tokens de distância).

### Armadilhas específicas do contexto longo

- **Acima de 262.144 o `llama-server` não serve.** Ele capa o slot ao contexto de treino
  (`tools/server/server-context.cpp:1202`) e ignora o YaRN — aloca o KV para o valor pedido e usa
  só 262k. Para contextos maiores use `llama-completion`.
- **YaRN é obrigatório acima de 262.144:** `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144`.
  A Qwen avisa que YaRN estático piora textos curtos, então é modo, não padrão.
- **Em 450k não cabe MTP** (~2 GiB). Em 262k com `q2_1` cabe, e é a melhor config de uso diário:
  `run-q2_1.ps1` dá 53,9 t/s com 262k, o que o `q4_0` não permitiria.
- O buffer de compute cresce ~4.096 B/token **a mais** quando o KV é quantizado, porque o caminho
  de prefill dequantiza uma camada por vez para f16. Isso vale para qualquer tipo quantizado e já
  está embutido nas contas de teto acima.


## Scripts

- **`start-server.bat [porta] [args extras]`** — o de sempre. 180k, KV q4_0, MTP n=2,
  `--host 0.0.0.0`, log em `logs/server-mtp.log`. `start-server.bat 11434` sobe na porta do ollama.
  Overrides por env: `CTX`, `NMAX`, `PORT`, `ALIAS`, `LLAMA_API_KEY`.
  O nome exposto na API é `qwen3.8-27b` (flag `-a/--alias`), não o caminho do `.gguf`.
- `run-200k.ps1` — llama-cli interativo, 200k, KV q4_0, sem spec. Baseline limpo.
- `run-200k-mtp.ps1 [-NMax 2]` — llama-server com MTP em `http://127.0.0.1:8080`, log em
  `logs/server-mtp.log`.
- `test-fullctx.ps1` — dispara o teste de contexto cheio pela API e grava `logs/fullctx-result.json`.

## Servir para outra máquina

- Endereço na LAN: `http://192.168.0.3:<porta>` (a Ethernet é perfil **Private**). O host também tem
  uma interface Radmin VPN, e `--host 0.0.0.0` expõe o server nela também — considerar ao decidir
  sobre API key.
- O firewall **já tem regra Allow inbound para `llama-server.exe`** (Private + Public), e ela é por
  programa, então vale para qualquer porta. Não precisa de admin nem de regra nova.
- Sem `LLAMA_API_KEY` o server é aberto: CORS `*` e nenhuma autenticação. Para exigir token:
  `set LLAMA_API_KEY=segredo & start-server.bat 11434` → clientes mandam `Authorization: Bearer segredo`.
### Claude Code apontando para cá

O llama-server serve `/v1/messages` (Anthropic Messages), e **function calling funciona** — testado:
`stop_reason: tool_use` com o input correto. `thinking` nos três modos (`adaptive`, `enabled`,
ausente) também é aceito, então não precisa de `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING`.

No `~/.claude/settings.json` da outra máquina (**não** no `.claude/` do projeto: em sessão
interativa o `env` de projeto só vale depois do wizard de primeira execução e do trust prompt):

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://192.168.0.3:11434",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_MODEL": "qwen3.8-27b",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.8-27b"
  }
}
```

**Template corrigido — `qwen35-tolerant.jinja`.** O template embutido no GGUF aborta com
`Jinja Exception: System message must be at the beginning.` se chegar uma mensagem `role: "system"`
depois da sequência inicial de systems (ele conta `num_sys` = systems no começo e dá
`raise_exception` em qualquer outra). O conversor Anthropic do llama.cpp repassa mensagens com o
role original (`tools/server/server-chat.cpp:377`), então um cliente que põe system dentro de
`messages` — e não só no campo `system` do topo — derruba toda requisição com **HTTP 500**.
A correção troca o `raise_exception` por emitir um turno `<|im_start|>system` normal; o ChatML
aceita system no meio sem problema. Verificado que tool calling continua funcionando depois disso.
O `.bat` já passa `--chat-template-file`.

**Quem faz isso é o próprio Claude Code** (confirmado no payload, v2.1.234): o prompt principal vai
no campo `system` do topo, mas a lista de agent types do Agent tool vai como uma **segunda mensagem
`role: "system"` dentro de `messages`**, depois da primeira mensagem do usuário — sequência
`system → user → system`. Sem o template corrigido, toda sessão do Claude Code morre em 500.

Para capturar payload de novo: `start-server.bat 11434 -lv 5`. **DEBUG é nível 5** — com `-lv 4` o
`SRV_DBG` do corpo não sai. Procure por `converted request:` no log (o corpo já convertido para o
formato OAI); o corpo cru Anthropic não é logado.

`ANTHROPIC_BASE_URL` é só a base, **sem `/v1`** — o Claude Code anexa `/v1/messages`.
`ANTHROPIC_AUTH_TOKEN` vai em `Authorization: Bearer`; `ANTHROPIC_API_KEY` iria em `x-api-key` e
exige aprovação única no modo interativo, por isso o token é o caminho mais curto.
`ANTHROPIC_SMALL_FAST_MODEL` está deprecado em favor de `ANTHROPIC_DEFAULT_HAIKU_MODEL`.

- **A porta 11434 não faz dele um ollama.** O llama-server só serve as rotas OpenAI
  (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`) e Anthropic (`/v1/messages`).
  As nativas do ollama (`/api/chat`, `/api/generate`, `/api/tags`) dão **404**. Cliente que fala
  OpenAI funciona só trocando a base URL; cliente em modo ollama, não. O campo `model` é ignorado,
  então pode mandar `qwen3.8:27b-mtp-q4_K_M` que ele aceita.
- `data/warpeace-700k.txt` — recorte de 700 KB = **166194 tokens**, para encher o contexto de verdade
  (o `warpeace.txt` inteiro dá ~900k tokens e estoura os 200k).
