# TurboQuant + llama.cpp — inferência de contexto longo

**Qwen3.8-27B numa única RTX 3090 (24 GB)**, com o maior contexto possível, inteiramente na GPU e
com velocidade utilizável. Este repositório é o registro do que foi medido, os scripts para
reproduzir, e as correções necessárias para chegar lá.

**Resultados:**

- **180k com KV `q4_0`**, o que já existia: 22,3 GB de VRAM, 60 t/s vazio, 31,5 t/s com 166k
  carregados, recall correto. O ponto de partida era 7 t/s e uma máquina travada.
- **428k com KV `q2_1`**, um tipo de 2,25 bpw que construímos para isso: prefill de 381 t/s e
  recall correto de informação a ~428 mil tokens de distância, ao custo de 4,86 % de perplexidade.

## Índice

- [Resultados](#resultados)
- [O que foi preciso descobrir](#o-que-foi-preciso-descobrir)
- [Reproduzir](#reproduzir)
- [Scripts](#scripts)
- [Servir na rede](#servir-na-rede)
- [KV em 2 bits: o tipo `q2_1`](#kv-em-2-bits-o-tipo-q2_1)
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
que o ganho. Medido com KV `q4_0` — e o ótimo depende do tipo de KV, ver
[Velocidade: o tipo de KV não é um eixo](#velocidade-o-tipo-de-kv-não-é-um-eixo).

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

O Qwen3.8-27B usa uma arquitetura híbrida, que o GGUF identifica como `qwen35` em
`general.architecture`: 65 blocos (64 + 1 de MTP), `full_attention_interval = 4` → **só 16 camadas
têm KV cache**; as outras 48 são SSM com estado constante que não cresce com o contexto.

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
| `start-server.bat [porta] [args]` | o de sempre: 180k, KV q4_0, MTP n=2, `--host 0.0.0.0`; YaRN automático acima de 262144 |
| `run-200k.ps1` | `llama-cli` interativo, 200k, sem speculative decoding — baseline limpo |
| `run-200k-mtp.ps1 [-NMax 2]` | server com MTP em `127.0.0.1:8080` |
| `test-fullctx.ps1` | teste de contexto cheio pela API: prefill, t/s e recall |
| `ab-perplexity.ps1` | A/B de qualidade do KV (f16 / q8_0 / q4_0 com e sem rotação) |
| `run-q2_1.ps1 [-Port] [-Ctx]` | server com KV `q2_1` em 262k **com MTP** — a config rápida de contexto grande |
| `run-longo.ps1 [-Ctx] [-Mtp]` | contexto longo com `q2_1` e YaRN: 327680 com MTP, ou 450560 sem |
| `ab-q2.ps1` | A/B de qualidade dos tipos de 2 bits |
| `bench-kv.ps1 [-Kvs] [-Ctx]` | q8_0/q4_0/q2_1 com MTP fixo, em duas profundidades de contexto; pré-voo de VRAM com `llama-fit-params` |
| `medir-modelo.ps1 -Model <gguf>` | mede split CUDA0/host, teto de contexto e perplexidade de um `.gguf` |
| `data/prepare-corpus.ps1` | recria o corpus de teste |
| `claude-local.cmd [args]` | abre o Claude Code contra o servidor local sem alterar o `settings.json` |
| `mcp-servers.json` | busca na web (DuckDuckGo via MCP) para a UI — ver `MCP` no `.bat` |
| `qwen35-tolerant.jinja` | template corrigido, necessário para o Claude Code |

Overrides do `.bat` por variável de ambiente: `CTX`, `KV`, `NMAX`, `PORT`, `ALIAS`, `MMPROJ`,
`LLAMA_API_KEY`. `NMAX=0` desliga o MTP, necessário acima de 327680.

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

Isso substitui o endpoint da Anthropic para **todas** as sessões daquela máquina: o Claude Code tem
um endpoint por sessão e não roteia por modelo — não existe chave de provider por modelo no
settings.json, e `ANTHROPIC_CUSTOM_MODEL_OPTION` só acrescenta um rótulo à lista do `/model`, sem
mudar para onde a requisição vai. Para manter os modelos da Anthropic intactos e usar o local só
quando quiser, use o `claude-local.cmd`, que exporta essas variáveis apenas para o processo filho.

**GitHub Copilot:** provider *Custom Endpoint*, API type *Chat Completions*, URL completa
`http://<ip>:11434/v1/chat/completions`, API key com qualquer texto.

Sem `LLAMA_API_KEY` o server é aberto na rede, com CORS `*` e nenhuma autenticação.

## Visão

O modelo é multimodal e o projetor está no mesmo repositório da Unsloth. Baixe e aponte:

```
curl -L -o E:/models/mmproj-F16.gguf \
  https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-F16.gguf

set MMPROJ=E:/models/mmproj-F16.gguf & start-server.bat 11434
```

Com `MMPROJ` definido o `.bat` desce o contexto para 128k sozinho, porque o projetor ocupa VRAM.
Nada muda no template — o do Qwen3.8 já trata imagem e vídeo.

### Custo medido

| item | custo |
|---|---|
| pesos do projetor (`mmproj-F16.gguf`) | **884,6 MiB** |
| pico extra ao processar imagem 1920×1080 | **105 MiB** |
| geração com visão ativa | 57,2 t/s — **inalterada** |
| total a 128k + visão | 23.220 MiB, ~1,3 GiB livres |

**A troca sai quase de graça.** Descer de 180k para 128k libera ~1 GiB de KV, que paga os 885 MiB do
projetor: o total de VRAM fica praticamente igual. Você troca 52k de contexto por visão.

Uma imagem custa quase nada de contexto. O projetor é `qwen3vl_merger` com `patch_size=16` e
`spatial_merge_size=2`, ou seja **1 token por bloco de 32×32 px**:

| imagem | tokens |
|---|---|
| 640×360 | 220 |
| 1024×1024 | 1024 |
| 1920×1080 | 1980 |

Medido: uma 1920×1080 deu 2108 tokens de prompt incluindo texto e template. O llama.cpp limita cada
imagem entre 8 e 4096 tokens; `--image-max-tokens N` aperta mais. Nos 128k cabem dezenas de imagens.

Verificado ponta a ponta nos dois formatos — `image_url` em `/v1/chat/completions` e bloco `image`
em `/v1/messages` — com respostas corretas sobre formas, cores, texto e contagem.

## KV em 2 bits: o tipo `q2_1`

O `q4_0` é o menor tipo de KV que o llama.cpp oferece, e com ele o teto nesta placa é
~295k tokens. Para passar disso construímos um tipo novo. O trabalho está na branch
`kv-q2_0` do clone do llama.cpp (não versionado aqui — veja [Reproduzir](#reproduzir)).

### O que já existia e não servia

O `GGML_TYPE_Q2_0` já existe no ggml, com bloco de 64 valores em 18 bytes = 2,25 bpw.
Faltava só o suporte a KV em CUDA, que implementamos. Os kernels passaram nos testes —
e o resultado foi **perplexidade 8,5422 contra 6,8594 do f16, uma regressão de 24,5 %**.

A causa não é a quantização em si, é o formato. Ele usa `d = amax` e mapeia o código `q`
para `(q-1)·d`. Como `|w| ≤ amax` por construção, o código `11` é inalcançável. Medido em
dados gaussianos:

| código | valor | uso |
|---|---|---|
| 00 | −d | 10,2 % |
| 01 | 0 | **79,6 %** |
| 10 | +d | 10,2 % |
| 11 | +2d | **0,0 %** |

Ele zera 80 % dos valores e desperdiça um quarto do espaço de código: gasta 2 bits para
carregar 1,58. Faz sentido para pesos ternários estilo BitNet, para os quais foi criado.
Para KV, destrói a informação.

### `q2_1`: mesmo tamanho, codebook correto

| | q2_0 | **q2_1** |
|---|---|---|
| escala | `amax` | `0,1510 × rms` |
| níveis | {−1, 0, +1, +2}·d | **{−10, −3, +3, +10}·d** |
| uso dos códigos | 10 / 80 / 10 / **0** % | 16,5 / 33,5 / 33,5 / 16,5 % |
| SNR | 2,88 dB | **9,41 dB** |

Os níveis inteiros são deliberados: o `vec_dot` do flash-attention usa `dp4a`, que exige
operandos int8. A razão 10/3 aproxima o ótimo de Lloyd-Max para gaussiana (1,5104/0,4528)
com erro de 0,1 %, então dá para ter o codebook quase ótimo **e** manter o produto escalar
inteiro. A rotação de Hadamard, que o llama.cpp já aplica, é o que torna os dados
gaussianos e justifica esse codebook.

### Qualidade medida

ctx 8192, 4 chunks, War and Peace, Qwen3.8-27B-Q4_K_M:

| KV | bpw | PPL | vs f16 |
|---|---|---|---|
| f16 | 16 | 6.8594 | — |
| q4_0 | 4,5 | 6.8701 | +0,16 % |
| q2_0 | 2,25 | 8.5422 | +24,5 % |
| **q2_1** | **2,25** | **7.1927** | **+4,86 %** |

### Velocidade: o tipo de KV não é um eixo

A tabela de qualidade acima é o preço do `q2_1`. Faltava saber se havia um segundo preço — ou um
segundo prêmio — no cronômetro: menos bits significa menos bytes lidos por passo, mas também mais
trabalho por byte para dequantizar. `bench-kv.ps1` mede os dois lados na mesma subida do servidor,
com MTP `n=4` constante para que só o tipo de KV varie.

ctx 98304, prefixo de 72.747 tokens, `Qwen3.8-27B-Q4_K_M`:

| KV | cache | curto (536 tok) | cheio (73k) | prefill | **ms/passo raso** | **ms/passo fundo** |
|---|---|---|---|---|---|---|
| q8_0 | 3413 MiB | 43,81 t/s | 30,53 t/s | 934 t/s | **55,67** | **74,11** |
| q4_0 | 1877 MiB | 42,59 t/s | 28,06 t/s | 928 t/s | **55,68** | **74,71** |
| q2_1 | 1013 MiB | 34,77 t/s | 28,77 t/s | 925 t/s | **55,85** | **76,22** |

As colunas de t/s sugerem que o tipo importa muito — 18 % de diferença em contexto curto. As duas
últimas mostram que não importa nada. Normalizando pelo **passo do modelo** em vez de por token
gerado (`draft_n / n_max` = número de forward passes), o custo de um passo é o mesmo nos três:
55,67 / 55,68 / 55,85 ms. Em contexto fundo o `q2_1` fica ~3 % mais lento, e o mesmo sinal aparece
num terceiro workload — o **contrário** do que a economia de bytes previa.

O kernel não é limitado por banda nessa profundidade. O `q8_0` lê 2,5 GiB de cache por passo e o
termo de atenção vale 18,4 ms: ~137 GB/s efetivos, uns 15 % do pico da 3090. Cortar os bytes pela
metade não compra nada, e a dequantização cobra a diferença. O prefill confirma: 1 % de
espalhamento entre 8,5 e 2,25 bpw.

**O que move o t/s é a aceitação de draft, e ela acompanha a precisão do KV** — 35,6 % / 34,0 % /
23,3 % em contexto curto. Os −18 % do `q2_1` são exatamente isso: 1,94 tokens por passo contra 2,37
do `q4_0`, razão 1,22, e a razão de t/s é 1,22. É a perplexidade +4,86 % aparecendo como
throughput, porque a cabeça MTP lê o mesmo cache grosseiro e rascunha pior.

Duas consequências. O tipo de KV se escolhe por VRAM e qualidade, como já vinha sendo feito — o
cronômetro não tem opinião. E o ótimo de `--spec-draft-n-max` é **por tipo de KV**: o `n=2` da
tabela de velocidade foi medido em `q4_0`, e com 23 % de aceitação o `q2_1` desperdiça verificação
demais em `n=4`.

### Contexto longo: 427.759 tokens

Com `q2_1` e YaRN (fator 4), prompt de 427.759 tokens numa única 3090:

| | |
|---|---|
| prefill | **381,6 t/s** (18,7 min) |
| geração | 9,5 t/s (sem MTP — não cabe neste contexto) |
| recall | **correto** |

A pergunta era onde a narrativa começa, com a resposta nas primeiras páginas — a ~428 mil
tokens de distância. O modelo respondeu São Petersburgo, soirée, Anna Pávlovna Schérer.
**KV em 2,25 bits preserva recall de longa distância.**

Ressalva: esse teste move duas variáveis ao mesmo tempo, `q2_1` e YaRN. O acerto sugere que
ambos estão bem, mas isolar exigiria repetir em 262k sem YaRN. E um probe único não é um
NIAH completo — a agulha estava no começo, a posição mais distante e também a mais fácil de
sondar.

### Tetos de contexto

Medidos com `llama-fit-params`, orçamento de 22.276 MiB (24.576 menos desktop e margem):

| pesos | KV q4_0 | KV q2_1 |
|---|---|---|
| Q4_K_M | ~295k | **~490k** |
| Q3_K_M | ~426k | ~700k |

O `llama-server` limitava o slot ao contexto de treino (`server-context.cpp:1202`) ignorando o
YaRN — alocava o KV para o valor pedido e usava só 262k. A branch corrige isso: o cap passa a
valer apenas quando não há RoPE scaling configurado. Com o patch, `run-longo.ps1` e o
`start-server.bat` servem contextos maiores pela interface web — ambos ligam
`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144` sozinhos acima de 262144.

O MTP custa ~1,5–2 GiB **fixos**, não proporcionais ao contexto: o contexto de draft
compartilha células com o alvo. Isso muda onde vale ligá-lo:

| config | geração (vazio) | VRAM livre | veredito |
|---|---|---|---|
| 450k + MTP | — | 319 MiB | **trava** — carrega, mas pagina durante a geração |
| 450k sem MTP | 36,8 t/s | ~860 MiB | ok |
| **327k + MTP** | **50,4 t/s** | **978 MiB** | **melhor dos dois** |

`run-longo.ps1 -Ctx 327680 -Mtp` ganha nos dois eixos: 37 % mais rápido que 450k sem MTP e
com mais folga. E como a geração cai conforme o contexto enche
(`ms/token ≈ 27,2 + 0,182 × preenchido/1000`, de 36,8 t/s vazio para 9,5 t/s em 428k), os
123k extras custariam caro justamente quando fossem usados.


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
