# TurboQuant — 200k de contexto num único RTX 3090

Rodar o **Qwen3.8-27B em 180k–200k tokens de contexto numa única RTX 3090 (24 GB)**, inteiramente na
GPU, com velocidade utilizável. Este repositório é o registro do que foi medido, os scripts para
reproduzir, e as correções necessárias para chegar lá.

**Resultado:** 180k de contexto estável a 22,3 GB de VRAM, **60 t/s** com contexto vazio e
**31,5 t/s** com 166k tokens carregados, com recall correto de informação a 166k tokens de
distância. O ponto de partida era 7 t/s e uma máquina travada.

## Índice

- [Resultados](#resultados)
- [O que foi preciso descobrir](#o-que-foi-preciso-descobrir)
- [Reproduzir](#reproduzir)
- [Scripts](#scripts)
- [Servir na rede](#servir-na-rede)
- [Estado do TurboQuant](#estado-do-turboquant)

## Resultados

Hardware: RTX 3090 24 GB · 32 GB RAM · Windows 11 · CUDA 13.2 · llama.cpp `d8df12e` (`sm_86`).
Modelo: `Qwen3.8-27B-Q4_K_M.gguf`, 15,93 GiB.

### Velocidade

| config | ctx | gen t/s | draft aceito |
|---|---|---|---|
| `llama-cli`, sem speculative decoding | 200k | 35,5 | — |
| `llama-server` + MTP `n=1` | 200k | 55,4 | 80 % |
| **`llama-server` + MTP `n=2`** | 200k | **60,1** | 70 % |
| `llama-server` + MTP `n=3` | 200k | 53,4 | 55 % |
| ollama (baseline anterior) | 64k | ~44 | — |

`--spec-draft-n-max 2` é o ótimo. A cabeça MTP prevê 1 token (`nextn_predict_layers=1`); o terceiro
token de rascunho derruba a aceitação de 70 % para 55 %, e a verificação desperdiçada custa mais
que o ganho.

### Contexto cheio — 166.312 tokens

| | |
|---|---|
| prefill | 255,8 s = **650 t/s** de média |
| geração | **31,5 t/s** (metade do vazio: cada token passa a ler ~3 GB de KV) |
| draft aceito | 74 % — **não** piora com contexto cheio |
| recall | **correto** |

O teste de recall pergunta onde a narrativa começa, com a resposta ~166k tokens atrás. O modelo
respondeu "São Petersburgo, soirée de Anna Pávlovna Schérer" — correto. **KV em 4 bits com rotação
de Hadamard preserva atenção de longa distância.**

Decaimento do prefill por posição no contexto:

| pos | 16k | 49k | 82k | 115k | 147k | 164k |
|---|---|---|---|---|---|---|
| t/s | 1113 | 853 | 674 | 557 | 461 | 427 |

Ajusta a `ms/token = 0,74 + 0,0098 × (pos/1000)`: custo constante de pesos/FFN/SSM mais um termo
**linear** na posição. Dez vezes mais contexto custa 2,6× mais tempo — num modelo denso seria ~10×.
É a arquitetura híbrida no cronômetro: só 16 das 64 camadas pagam atenção quadrática.

## O que foi preciso descobrir

### 1. Por que 200k parecia impossível

O modelo é `qwen35`: 65 blocos (64 + 1 de MTP), `full_attention_interval = 4` → **só 16 camadas têm
KV cache**; as outras 48 são SSM com estado constante que não cresce com o contexto.

KV por token = 16 camadas × 4 heads_kv × (256+256) = 32768 elementos. Medido com
`llama-fit-params --fit-print on` (MiB em CUDA0, `-fa on`):

| KV | ctx | modelo | KV | compute | total | cabe em 24 GB? |
|---|---|---|---|---|---|---|
| f16 | 200k | 15345 | 12661 | 505 | 28511 | não |
| q8_0 | 200k | 15345 | 6796 | 1057 | 23198 | **não** |
| q4_0 | 200k | 15345 | 3668 | 1057 | 20070 | sim |
| q4_0 | 262k | 15345 | 4757 | 1360 | 21462 | sim — contexto nativo inteiro |

**q8_0 não fecha em 200k; 4 bits fecha, inclusive nos 262144 nativos.** Por isso a qualidade do KV
em 4 bits é *o* problema deste projeto, não a memória.

### 2. O modo de falha não é OOM

Com `--fit on` (padrão) o llama.cpp **não falha** quando a config não cabe: ele reduz o contexto ou
empurra camadas para a CPU. O sintoma é velocidade — 7 t/s em vez de 35 — e pressão de RAM até o
sistema travar.

E mesmo cabendo na GPU, encher a placa trava a máquina de outro jeito: a 23,9 GB de 24,5 o desktop
fica sem VRAM, o WDDM começa a paginar VRAM para a RAM e a máquina inteira engasga, **sem escrever
uma linha de erro no log**. Tudo é pré-alocado, então o pico durante a geração fica ~30 MiB acima do
idle: se carregou com folga, não engasga depois.

| config | pico de VRAM | livre | veredito |
|---|---|---|---|
| 200k + MTP | 23900 MiB | ~600 MiB | trava o PC |
| **180k + MTP** | 22780 MiB | ~1800 MiB | ok com o VS Code fechado ← padrão |
| 128k + MTP | ~21300 MiB | ~3200 MiB | folgado |

**Regra:** sempre `-ngl 99 -fit off`, para que uma config que não cabe morra com CUDA OOM em
segundos em vez de escorrer para a RAM. E mirar ≥1,5 GiB livres, não zero.

### 3. Duas armadilhas de configuração

- **MTP só existe no `llama-server`** — o `llama-cli` nem cria o contexto de draft. Não precisa de
  segundo modelo: `--spec-type draft-mtp` cria um contexto `LLAMA_CONTEXT_TYPE_MTP` sobre os mesmos
  pesos, a ~1,8 GiB de custo.
- **O KV do contexto de draft nasce em f16.** Sem `-ctkd q4_0 -ctvd q4_0` ele tenta alocar KV f16 em
  cima dos 200k. E o server abre **4 slots** por padrão: com 4, sobram 317 MiB livres. Use `-np 1`.

### 4. O template do Qwen quebra o Claude Code

O template embutido no GGUF conta `num_sys` = mensagens `system` no começo e dá `raise_exception`
em qualquer system depois disso. O Claude Code manda o prompt principal no campo `system` do topo,
mas injeta a lista de agent types como uma **segunda mensagem `role: "system"` dentro de
`messages`** — sequência `system → user → system`. Como o conversor Anthropic do llama.cpp repassa
o role original (`tools/server/server-chat.cpp:377`), **toda** requisição morria em HTTP 500.

`qwen35-tolerant.jinja` troca o `raise_exception` por emitir um turno `<|im_start|>system` normal.
ChatML aceita system no meio sem problema — a proibição era escolha do template. Tool calling
continua funcionando.

## Reproduzir

**1. llama.cpp com CUDA** (não incluso aqui; é um clone separado em `llama.cpp/`):

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
```

`CMAKE_CUDA_ARCHITECTURES=86` é a 3090; ajuste para sua placa.

**2. Modelo:** `Qwen3.8-27B-Q4_K_M.gguf` (quantização da Unsloth). Ajuste o caminho em
`start-server.bat`.

**3. Corpus de teste:** `data/prepare-corpus.ps1` baixa War and Peace do Project Gutenberg e corta o
recorte de 700 KB = 166.194 tokens.

**4. Antes de qualquer teste, estime** — não carrega o modelo, é instantâneo:

```powershell
.\llama-fit-params.exe -m <modelo> -c 200000 -ctk q4_0 -ctv q4_0 -fa on --fit-print on
```

Saída: `dispositivo modelo contexto compute` em MiB. Some e compare com sua VRAM livre.

## Scripts

| script | o que faz |
|---|---|
| `start-server.bat [porta] [args]` | o de sempre: 180k, KV q4_0, MTP n=2, `--host 0.0.0.0` |
| `run-200k.ps1` | `llama-cli` interativo, 200k, sem speculative decoding — baseline limpo |
| `run-200k-mtp.ps1 [-NMax 2]` | server com MTP em `127.0.0.1:8080` |
| `test-fullctx.ps1` | teste de contexto cheio pela API: prefill, t/s e recall |
| `ab-perplexity.ps1` | A/B de qualidade do KV (f16 / q8_0 / q4_0 com e sem rotação) |
| `data/prepare-corpus.ps1` | recria o corpus de teste |
| `qwen35-tolerant.jinja` | template corrigido, necessário para o Claude Code |

Overrides do `.bat` por variável de ambiente: `CTX`, `NMAX`, `PORT`, `ALIAS`, `LLAMA_API_KEY`.

## Servir na rede

`start-server.bat 11434` sobe em `0.0.0.0` na porta do ollama. O nome exposto na API é o do
`--alias`, não o caminho do `.gguf`.

**Rotas disponíveis:** OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`) e
Anthropic (`/v1/messages`). As nativas do ollama (`/api/chat`, `/api/tags`) dão **404** — usar a
porta 11434 não faz dele um ollama. O campo `model` é ignorado: qualquer nome é servido pelo modelo
carregado.

**Claude Code** (`~/.claude/settings.json` da outra máquina, não o `.claude/` do projeto):

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://<ip-do-host>:11434",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_MODEL": "qwen3.8-27b",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.8-27b"
  }
}
```

A base URL vai **sem `/v1`** (o Claude Code anexa `/v1/messages`). Use `ANTHROPIC_AUTH_TOKEN`, que
vai em `Authorization: Bearer` e vale na hora — `ANTHROPIC_API_KEY` vai em `x-api-key` e exige uma
aprovação única no modo interativo. Function calling e `thinking` nos três modos foram verificados.

**GitHub Copilot:** provider *Custom Endpoint*, API type *Chat Completions*, URL completa
`http://<ip>:11434/v1/chat/completions`, API key com qualquer texto.

Sem `LLAMA_API_KEY` o server é aberto na rede, com CORS `*` e nenhuma autenticação.

## Estado do TurboQuant

O estágio 1 do TurboQuant — rotação de Walsh-Hadamard antes de quantizar o KV — **já está no
upstream do llama.cpp**. Ela liga sozinha quando o KV é quantizado e `head_dim % 64 == 0` (aqui é
256). Confirme com `attn_rot_k = 1` no log de carga. `LLAMA_ATTN_ROT_DISABLE=1` desliga, e é o A/B
para medir o ganho.

Não reimplemente isso. A pergunta era o estágio 2 — um quantizador escalar melhor que o
round-to-nearest com escala por bloco de 32 do `q4_0` — e ela foi respondida por medição
([`ab-perplexity.md`](ab-perplexity.md), ctx 32k, 8 chunks):

| config | PPL | Δ vs f16 |
|---|---|---|
| f16 (referência) | 7.9887 | — |
| q8_0 + rotação | 7.9909 | +0,028 % |
| **q4_0 + rotação** | **7.9998** | **+0,139 %** |
| q4_0 SEM rotação | 8.0059 | +0,215 % |

Quantizar o KV de f16 para q4_0 — 3,5× menos memória, de 12,4 para 3,6 GiB aos 200k — custa
**0,139 %** de perplexidade. A rotação de Hadamard entrega ~35 % dessa economia de dano (0,215 % →
0,139 %). E o que sobra para um quantizador melhor é justamente esses 0,139 %, no melhor caso
absoluto.

**Veredito: não abrir o fork.** 0,139 % não paga um tipo ggml novo, kernels CUDA e suporte em
flash-attention, muito menos mantê-los rebaseando contra uma árvore que se move dezenas de commits
por dia. O objetivo — 200k numa única 3090 — foi atingido com o que já existe no upstream.

Ressalvas em [`ab-perplexity.md`](ab-perplexity.md): as barras de erro são maiores que os deltas
(a comparação é pareada e a ordem saiu monotônica, mas falta o delta chunk a chunk), e a medição foi
a 32k, não a 180k.

### Se for mexer no llama.cpp

Fork no GitHub com branch que rebaseia, **não** fork divergente. O clone aqui é `--depth 1` e ficou
5 commits atrás poucas horas depois do fetch — essa é a velocidade que um fork paralelo teria que
absorver para sempre, e tudo que deu os 60 t/s (MTP, `fit-params`, a própria rotação) é trabalho
upstream recente. Comece com `git fetch --unshallow`; sem histórico não há rebase nem bisect.
