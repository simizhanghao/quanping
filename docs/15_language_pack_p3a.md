# P3 Multilingual — LanguagePack Plan & P3-A Contract

**Principle:** LanguagePack 是组织协议；Benchmark 是插件。不造第二个 scorer / compare / bootstrap。

## Slice plan

| Slice | Scope | Status |
|-------|--------|--------|
| **A** | `LanguageSpec` / `BenchmarkSpec` / `LanguagePackSpec` + Registry + inspect CLI | ← now |
| **B** | Parallel reading (Belebele) via adapter；`ind` + 第二语言 | planned |
| **C** | Native Indonesian capabilities（IndoMMLU → COPAL-ID） | planned |
| **D** | External executor adapter（lm-eval + `--log_samples` → PredictionRecord） | planned |
| **E** | Cross-language × capability Base↔Candidate regression report + gates | planned |

## P3-A objects

```text
LanguageSpec     iso639_3 / script / macrolanguage / variant
BenchmarkSpec    capability + task_type + language + provenance + version/revision + status
LanguagePackSpec capabilities: {reading_comprehension: [benchmark_ids], ...}
```

Provenance must declare `origin` + `translation` (`native` | `parallel` | `human_translated` | …).

Unavailable benchmarks stay `NOT_AVAILABLE` — **never fill 0**.

## CLI

```text
linguaeval language-inspect-offline <config.yaml>
  → language_pack_audit.json / report.md
```

## Hard acceptance (P3-A)

1. Register `ind` + `arb` (arb may set macrolanguage=`ara`) — Kernel 无语言 if/else  
2. Unknown language → `NOT_AVAILABLE` / registry error — **no English fallback**  
3. Swap pack only via config (`ind_v1` ↔ `arb_v1`)  
4. BenchmarkSpec requires capability / task_type / language / provenance / version|revision  
5. Native vs translated/parallel distinguishable  
6. Pack benchmark stub `NOT_AVAILABLE` without inventing scores  
7. No changes to Scorer / Comparator / Statistics / Gate core for this slice  

## Non-goals (this slice)

Belebele download、IndoMMLU、lm-eval、multilingual 总分、自动翻译、LLM Judge。
