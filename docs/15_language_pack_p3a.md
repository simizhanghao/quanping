# P3 Multilingual — LanguagePack Plan & P3-A Contract

**Principle:** LanguagePack 是组织协议；Benchmark 是插件。不造第二个 scorer / compare / bootstrap。

## Slice plan

| Slice | Scope | Status |
|-------|--------|--------|
| **A** | `LanguageSpec` / `BenchmarkSpec` / `LanguagePackSpec` + Registry + inspect CLI | ✅ |
| **B** | Parallel reading (Belebele-format) via adapter；`ind` + `arb` | ✅ (toy fixtures) |
| **C** | Native Indonesian capabilities（IndoMMLU → COPAL-ID） | ✅ (toy fixtures) |
| **D** | External executor adapter（lm-eval MC `--log_samples` → PredictionRecord） | ✅ |
| **E** | Cross-language × capability Base↔Candidate regression report + gates | ✅ |
| **F** | Real-data & regression hardening（metric_path + P1 compare + encoding + real Base/SFT） | ✅ (F-D: run `scripts/run_language_real_p3fd.sh` for numbers) |

**P3-A～E** = 框架能力闭环（接口正确）。**P3-F** = 结论可信封板（统计 + 真实数据路径）；跑通 F-D 后再进 P4。无 multilingual total score。

## P3-F — Real-Data & Regression Hardening

完成标准（一句）：

> Language evaluation 不再特殊使用 accuracy-only / delta-only 语义，并在真实 Base/SFT + 真实 multilingual benchmark 上产出 statistically supported capability regression report。

| Sub-slice | Scope | Status |
|-----------|--------|--------|
| **F-A** | `report.primary_target` / `primary_metric` → `metric_path` / `baseline_value` / `candidate_value` / `delta`（禁 `delta_accuracy` 硬编码） | ✅ |
| **F-B** | Language regression **复用** P1 Comparator（`sample_id` 对齐 + Gain/Regression + paired bootstrap CI）；Gate 走 `ci_low` + `min_samples` / `INSUFFICIENT_SUPPORT` | ✅ |
| **F-C** | Native MC 显式 `answer_encoding`（`letter` / `zero_based_index` / `one_based_index`）；`lm_eval_samples` / `lm_eval_mc_samples` 契约标 **MC-only** | ✅ |
| **F-D** | Real Base vs SFT：Belebele-ID + IndoMMLU + COPAL-ID + control Belebele-eng；prepare + predict + matrix config | ✅ scaffolding (run locally for numbers) |

### F-A regression schema

```yaml
report:
  primary_target: answer
  primary_metric: accuracy   # MetricSpec 决定比较什么；matrix 不假定 accuracy
statistics:
  enabled: true
  n_bootstrap: 200
  bootstrap_unit: sample
```

```text
language_regression.by_capability.<cap>.by_language.<lang>:
  engine: p1_paired_compare
  metric_path / baseline_value / candidate_value / delta
  delta_ci_low / delta_ci_high
  transitions / statistics / support
```

### F-B paired discipline

```text
LanguagePack cell
  → baseline + candidate PredictionRecords
  → linguaeval.compare.paired.compute_paired_comparison  (same as compare-offline)
  → Gain / Regression / Δ / 95% CI
  → language × capability report + gates
```

Do **not** invent a second language-only regression. Point-estimate gates may still use `.delta`; CI gates use `.delta_ci_low` + `requirements.min_samples`.

### F-C answer encoding + lm-eval MC-only

```yaml
# IndoMMLU letters
answer_encoding: letter
# COPAL-ID label 0/1
answer_encoding: zero_based_index
# Belebele correct_answer_num (official one-indexed) — adapter-specific
answer_encoding: one_based_index
```

`lm_eval_mc_samples` (= `lm_eval_samples`) imports **MC log_samples only**. Generation dumps must wait for a separate adapter.

### F-D real-data matrix（3+1，不贪多）

| Capability | Benchmark | Role |
|------------|-----------|------|
| Ind Reading | Belebele-ID (`ind_Latn`) | target parallel reading |
| Ind Knowledge | IndoMMLU | native knowledge |
| Ind Culture | COPAL-ID | native cultural reasoning |
| Control Reading | Belebele-eng (`eng_Latn`) | 检测印尼业务 SFT 是否伤及其他语言 |

产出表：`Base / SFT / Δ / 95% CI` × language×capability。

```bash
# prepare subset + predict Base/SFT + language-matrix-offline
bash scripts/run_language_real_p3fd.sh
# artifacts → results/27_language_capability_real_base_sft/
```

Docs: `examples/language_pack_real/README.md` · Config: `configs/examples/27_language_capability_real_base_sft.yaml`

### After P3-F → P4 order（不先 Safety）

```text
P4-A OpenAI-compatible ModelAdapter (vLLM endpoint)
P4-B Production Profiler (TTFT/TPOT/E2E p50–p99 / goodput / VRAM)
P4-C SafetyPack (Unsafe Compliance / Over-refusal / Safe Completion；IndoSafety via LanguagePack)
P4-D Unified Release Card / Gate（仍无万能总分）
```

## P3-E Capability regression report + gates

```text
language-matrix-offline (+ gates:)
  → language_metrics.json
  → language_regression.json
  → language_capability_report.json   # rows: language × capability
  → gate.json                         # reuse compare GateEngine
  → report.md                         # markdown matrix (no total score)
```

Example gates (point + CI with P1-D support policy):

```yaml
gates:
  - id: target_ind_reading_min_gain
    path: language_regression.by_capability.reading_comprehension.by_language.ind.delta
    op: ">="
    value: 0.0
  - id: other_arb_reading_max_drop
    path: language_regression.by_capability.reading_comprehension.by_language.arb.delta
    op: ">="
    value: -0.05
  - id: target_ind_delta_ci_lower
    path: language_regression.by_capability.reading_comprehension.by_language.ind.delta_ci_low
    op: ">="
    value: 0.0
    requirements:
      min_samples: 500
```

```bash
PYTHONPATH=src python -m linguaeval language-matrix-offline \
  configs/examples/26_language_capability_report_gates_toy.yaml
```

## P3-D lm-eval **MC** samples adapter

```text
lm-eval --log_samples  (external, multiple-choice only)
      ↓
samples JSON / JSONL
      ↓
adapter: lm_eval_mc_samples  (= lm_eval_samples)
      + answer_encoding: letter|zero_based_index|one_based_index
      ↓
SampleRecord + PredictionRecord
      ↓
score-offline / language-matrix-offline  (existing Kernel)
```

Hard rules: **no `lm_eval` Python dependency** in LinguaEval; never trust dump `acc` — re-score via ScoreRecord; **MC-only** (no generation).

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
