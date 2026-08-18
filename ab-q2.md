# KV q2_0 - portao de decisao da Fase 1

Modelo: Qwen3.8-27B-Q4_K_M.gguf | ctx 8192 | 4 chunks | KV na CPU (-nkvo)

| KV | bpw | PPL | +/- | delta vs f16 | min |
|---|---|---|---|---|---|
| f16 | 16 | 6.8594 | 0.12382 | 0,000% | 1.3 |
| q4_0 | 4.5 | 6.8673 | 0.12402 | 0,115% | 0.9 |
| q2_0 | 2.25 | FALHOU | - | - | 12.7 |
