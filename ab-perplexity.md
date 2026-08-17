# A/B de qualidade do KV cache

Qwen3.8-27B-Q4_K_M · ctx 32768 · 8 chunks · War and Peace · RTX 3090 · ~5 min por config.
Gerado por [`ab-perplexity.ps1`](ab-perplexity.ps1).

| config | PPL | ± | Δ vs f16 |
|---|---|---|---|
| f16 (referência) | 7.9887 | 0.0532 | — |
| q8_0 + rotação | 7.9909 | 0.0533 | +0.028 % |
| **q4_0 + rotação** ← o que rodamos | **7.9998** | 0.0533 | **+0.139 %** |
| q4_0 SEM rotação | 8.0059 | 0.0534 | +0.215 % |

Validade do ablation: o aviso `attention rotation force disabled` aparece apenas no log da run
`q4_0 SEM rotação`, confirmando que `LLAMA_ATTN_ROT_DISABLE=1` fez efeito só onde devia.

## Leitura

**Quantizar o KV custa quase nada neste modelo.** Ir de f16 para q4_0 — 3,5× menos memória de KV,
de 12,4 GiB para 3,6 GiB aos 200k — piora a perplexidade em **0,139 %**. É o que torna os 200k
possíveis numa 3090, e sai praticamente de graça.

**A rotação de Hadamard entrega cerca de um terço disso.** Sem ela o dano é 0,215 %; com ela,
0,139 %. Ou seja, o estágio 1 do TurboQuant recupera ~35 % da degradação de quantizar em 4 bits.
Funciona, e já vem ligado sozinho no upstream.

**O prêmio máximo do estágio 2 é 0,139 %.** Essa é a distância que sobra entre `q4_0 + rotação` e o
f16 sem perda. Um quantizador perfeito — melhor que qualquer coisa publicada — recuperaria isso e
nada mais. Um quantizador realista recuperaria talvez metade: ~0,07 %.

## Ressalvas honestas

**As barras de erro são maiores que os deltas.** O ± 0,053 é o erro padrão entre chunks; o maior
delta é 0,017. Tomadas isoladamente, essas diferenças não são estatisticamente distinguíveis. O que
sustenta a leitura é que (a) as quatro runs usam exatamente os mesmos chunks, então a comparação é
pareada e o erro entre chunks se cancela, e (b) a ordem saiu monotônica e exatamente na direção que
a teoria prevê — f16 < q8_0 < q4_0+rot < q4_0−rot. Isso é sinal, não sorte. Mas para afirmar
significância seria preciso o delta pareado chunk a chunk, que este experimento não coletou.

**Medido a 32k, não a 180k.** O erro de quantização do KV se acumula com o comprimento do contexto,
então a degradação no regime que o projeto realmente usa pode ser maior. Para inverter a conclusão,
porém, ela teria que crescer cerca de dez vezes.

**Perplexidade não é recall.** Ela mede previsão local. O teste de contexto cheio
([`test-fullctx.ps1`](test-fullctx.ps1)) cobre o outro lado: com 166.312 tokens carregados, o modelo
respondeu corretamente uma pergunta sobre as primeiras páginas.

## Veredito

**Não abrir o fork.** O estágio 1 já está no upstream e o que sobra para o estágio 2 é 0,139 % de
perplexidade — não paga escrever um tipo ggml novo, kernels CUDA de quantização/dequantização e
suporte no flash-attention, nem mantê-los rebaseando contra uma árvore que se move dezenas de
commits por dia.

O objetivo do projeto — 200k de contexto numa única 3090 — foi atingido com o que já existe.
