# P3 Multilingual — LanguagePack Plan & P3-A Contract

**Principle:** LanguagePack 是组织协议；Benchmark 是插件。不造第二个 scorer / compare / bootstrap。

## Slice plan

| Slice | Scope | Status |
|-------|--------|--------|
| **A** | `LanguageSpec` / `BenchmarkSpec` / `LanguagePackSpec` + Registry + inspect CLI | ✅ |
| **B** | Parallel reading (Belebele-format) via adapter；`ind` + `arb` | ✅ (toy fixtures) |
| **C** | Native Indonesian capabilities（IndoMMLU → COPAL-ID） | ✅ (toy fixtures) |
| **D** | External executor adapter（lm-eval + `--log_samples` → PredictionRecord） | ✅ |
| **E** | Cross-language × capability Base↔Candidate regression report + gates | ✅ |

**P3 frozen** at E: LanguagePack protocol + parallel/native fixtures + lm-eval import path + gated capability report. No multilingual total score.

## P3-E Capability regression report + gates

```text
language-matrix-offline (+ gates:)
  → language_metrics.json
  → language_regression.json
  → language_capability_report.json   # rows: language × capability
  → gate.json                         # reuse compare GateEngine
  → report.md                         # markdown matrix (no total score)
```

Example gates:

```yaml
gates:
  - id: target_ind_reading_min_gain
    path: language_regression.by_capability.reading_comprehension.by_language.ind.delta_accuracy
    op: ">="
    value: 0.0
  - id: other_arb_reading_max_drop
    path: language_regression.by_capability.reading_comprehension.by_language.arb.delta_accuracy
    op: ">="
    value: -0.05
```

```bash
PYTHONPATH=src python -m linguaeval language-matrix-offline \
  configs/examples/26_language_capability_report_gates_toy.yaml
```

## P3-D lm-eval samples adapter

```text
lm-eval --log_samples  (external)
      ↓
samples JSON / JSONL
      ↓
adapter: lm_eval_samples
      ↓
SampleRecord + PredictionRecord
      ↓
score-offline / language-matrix-offline  (existing Kernel)
```

Hard rules: **no `lm_eval` Python dependency** in LinguaEval; never trust dump `acc` — re-score via ScoreRecord.

```bash
PYTHONPATH=src python -m linguaeval score-offline \
  configs/examples/25_score_lm_eval_samples_toy.yaml
```

## P3-C Native Indonesian capabilities

```text
indommlu_jsonl  → local_knowledge   (native_authored)
copal_jsonl     → cultural_reasoning (native_authored, culture_sensitive)
belebele_jsonl  → reading_comprehension (parallel; contrast)
      ↓
language-matrix-offline capabilities: {...}
      → language_metrics.json   # by_capability × by_language
      → language_regression.json
```

No multilingual total score. Full IndoMMLU/COPAL downloads out of this slice.

```bash
PYTHONPATH=src python -m linguaeval language-matrix-offline \
  configs/examples/24_language_capability_ind_native_toy.yaml
```

## P3-B Belebele-format path

```text
belebele_jsonl adapter
      ↓
SampleRecord (gold.answer ∈ {A,B,C,D}) + meta.language
      ↓
shared TaskSpec / MetricSpec / score_targets
      ↓
language-matrix-offline
      → language_metrics.json
      → language_regression.json   # candidate − baseline per language
```

Hard rule: **same adapter** for `ind` and `arb`; only `source.language` + data path change. Full Belebele download is out of this slice (toy parallel fixtures only).

```bash
PYTHONPATH=src python -m linguaeval language-matrix-offline \
  configs/examples/22_language_matrix_belebele_toy.yaml
```

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
