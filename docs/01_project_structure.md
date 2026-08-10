# LinguaEval — Project Structure Arrangement

**Brand:** LinguaEval — Business-Aware Evaluation Harness for Multilingual SFT Models  
**Repo dir:** `eval_factory/`（对内目录名；对外品牌用 LinguaEval）  
**Doc status:** P0 structure freeze candidate  
**Date:** 2026-08-10  

---

## 0. One-line Positioning

> **架构从第一天通用；实现从第一天收敛。**  
> N2S 是第一个复杂 Reference Case，不是 Kernel。

不是 HELM 复刻，也不是 N2S scorer 重构，而是：

**小语种业务 SFT 的回归评测协议 + 可插拔 Pack 系统。**

核心创新四点：

1. **Business-aware** — 任意业务 schema / multi-target task  
2. **Multilingual-aware** — Language Pack 按 ISO 639-3 装配  
3. **Regression-aware** — 默认 Base↔SFT 比较（D3 是算子，不是单独 benchmark）  
4. **Reliability-aware** — robustness / calibration / context / consistency / efficiency  

---

## 1. Design Principles（硬约束）

| # | Principle | Meaning |
|---|-----------|---------|
| P1 | Global contracts first | Sample/Task/Output/Metric/Model/Prediction/Manifest 一次设计正确 |
| P2 | Convergent implementation | v0.1 不实现十维，但十维都有 Contract 状态 |
| P3 | No Empty Pack Rule | 未满足 5 条件不得标 `supported` |
| P4 | Deterministic > Judge | 分类/JSON/约束优先 rule-based scorer |
| P5 | Offline-scoreable | 改 Metric 不应强制重跑模型（PredictionRecord 解耦） |
| P6 | Applicability explicit | 不适用写 `NOT_APPLICABLE`，禁止假 0 分 |
| P7 | Comparability protocol | 效率/延迟仅同 backend×sampling×max_tokens×concurrency 可比 |
| P8 | N2S is Example-01 | Kernel 不得出现 `n2s`/`routing_skill` 硬编码字段名作为唯一路径 |
| P9 | Markdown report first | HTML dashboard 延后；CI gate 可先 JSON |
| P10 | Inspect/lm-eval optional | 不进 Core dependency |

### No Empty Pack Rule（5 条件）

一个维度/Pack 只有同时具备以下 5 项，才可在 README 标 `supported`：

1. **Contract** — 输入/输出/applicability  
2. **Implementation** — ≥1 真实实现  
3. **Smoke Dataset** — 32–64 条可跑样本  
4. **Artifact** — 固定结果文件名  
5. **Test/Gate** — 自动化验收  

否则状态必须是 `planned`。

---

## 2. Evaluation Taxonomy（D0 + D1–D10）

工程上按 **5 层**组织，而不是十个平级 benchmark：

```text
                 Evaluation System
                        │
              D0 Evaluation Integrity
               “这个实验可信么？”
                        │
       ┌────────────────┼──────────────────┐
       ↓                ↓                  ↓
   Task Layer      Reliability Layer    System Layer
       │                │                  │
  D1 Business      D5 Robustness      D10 Efficiency
  D2 Schema/IF     D6 Context
  D4 Language      D7 Calibration
  D9 Safety        D8 Consistency
       │                │
       └──────── D3 Base↔SFT Regression ────────┘
                 （横切比较算子）
```

| ID | Name | 通用对象 | 本质 | v0 状态目标 |
|----|------|----------|------|-------------|
| D0 | Integrity | `DataAuditor` | 实验可信度 | P0/P1 **必做** |
| D1 | Business / Task | `TaskScorer` | 任务效果 | P1 **必做** |
| D2 | Schema / IF | `OutputValidator` | 格式与约束 | P1 **必做** |
| D3 | Regression | `ModelComparator` | 任意 metric 的 Δ | P1 **必做** |
| D4 | Language | `LanguagePack` | 目标语能力 | P3 planned |
| D5 | Robustness | `PerturbationEngine` | clean vs perturbed | P2 planned |
| D6 | Context | `ConversationRunner` | context strategy | P2 planned |
| D7 | Calibration | `ConfidenceEvaluator` | ECE/AURC/决策曲线 | P1.5 planned |
| D8 | Consistency | `Repeat/MetamorphicRunner` | agreement / flip | P2/P3 planned |
| D9 | Safety | `SafetyPack` | unsafe / over-refusal | P4 planned |
| D10 | Efficiency | `Profiler` | TTFT/p95/VRAM | P1 telemetry → P4 完整 |

**不适用规则：** Pack 必须声明 `applicability`；report 对不适用项输出 `NOT_APPLICABLE + reason`，禁止填 0。

---

## 3. Core Contracts（P0 必须冻结的 7 个）

> 字段级 JSON Schema 另文冻结（见 `02_core_contracts.md`）。  
> 本文只固定**职责边界与最小必填面**。

### 3.1 `SampleRecord` — 样本统一层

- **不知道**什么叫 N2S。  
- `gold` 为开放 dict，由 `TaskSpec.targets.*.path` 解释。  
- `conversation` **可选**；无对话任务为 `null` / 省略。

最小面：

```text
sample_id
input.{text|messages}
gold: object
meta.{language, domain, source, split, ...}
conversation?: {dialogue_id, turn_id, role, context_mode, ...}
```

### 3.2 `TaskSpec` — 如何解释样本

- `task_type`: classification | structured_multitask | extraction | qa | generation | …  
- `targets[]`: name / type / path / condition / labels  
- **Multi-target + conditional targets** 是一等能力（N2S 的 n2s/routing/intent 只是一种实例化）

### 3.3 `OutputSpec` — 模型输出形态（与 Task 分离）

- `format`: json | text | xml | …  
- `schema` / `parser` / `constraints`（no_markdown、enum、language…）  
- 同一 Task 可挂不同 OutputSpec。

### 3.4 `MetricSpec` — 算什么分

- 按 target / joint 声明 metric 列表与参数（如 Fβ 的 β）  
- 不决定 inference。

### 3.5 `ModelSpec` — 跑谁、怎么跑

- `backend` / `model` / `template` / sampling / `max_tokens` / concurrency  
- `comparability_group`：D3/D10 比较前程序校验  
- 典型键：`base`, `sft`（可扩展更多）

### 3.6 `PredictionRecord` — 推理与打分解耦的关键中间层

每模型 × 每样本一条：

```text
sample_id, model_id
raw_output
parsed
scores / confidences   # 可供 D7
format.{parse_ok, schema_ok, ...}
usage.{prompt_tokens, completion_tokens}
timing.{latency_ms, ttft_ms}
error
```

**改 MetricSpec / 重跑 Scorer 默认不重新推理。**

### 3.7 `RunManifest` — 一次 run 的可复现元数据

```text
run_id, created_at, git_sha (optional)
config_hash
models, task, output, metrics, packs_requested
data_fingerprint
env.{gpu, backend_versions}
artifact_index
gate_result?
```

结果目录用编号英文名（见 §6），timestamp 只进 manifest，不进目录主键。

---

## 4. Kernel Pipeline

```text
DatasetAdapter
     ↓
 SampleRecord
     ↓
 TaskSpec (+ OutputSpec)
     ↓
 PromptBuilder          ← P1 完整；P0 可跳过（offline-only）
     ↓
 ModelAdapter           ← P1；P0 可跳过
     ↓
 PredictionRecord
     ↓
 Parser / OutputValidator   ← D2
     ↓
 TaskScorer                 ← D1
     ↓
 Aggregator (+ Slices)
     ↓
 Reporter (Markdown) + GateEngine
```

### Side plugins（按 Pack 挂载，非 Core 阻塞）

| Plugin | Dimension |
|--------|-----------|
| `DataAuditor` | D0 |
| `ModelComparator` | D3 |
| `PerturbationEngine` | D5 |
| `ConversationRunner` | D6 |
| `ConfidenceEvaluator` | D7 |
| `Repeat/MetamorphicRunner` | D8 |
| `LanguagePack` | D4 |
| `SafetyPack` | D9 |
| `Profiler` | D10 |
| `SlicePlugin` | diagnostics（固定 slice 优先于 auto-discovery） |

**可选外部适配（非 Core）：**

```text
LinguaEval Core
├── Native: vLLM / HF / OpenAI-compatible
├── Optional: lm-eval adapter
└── Optional: Inspect adapter
```

---

## 5. Repository Layout（目标结构）

> P0 **只创建文档与最小可运行 offline kernel 所需路径**；  
> 禁止预先创建一堆空 `safety/`、`robustness/` 实现文件并标 supported。

```text
eval_factory/                         # repo root（品牌 LinguaEval）
├── README.md                         # supported / planned 两表
├── pyproject.toml                    # 稍后
├── docs/
│   ├── 00_executive_brief.md         # 立项 / 汇报口径
│   ├── 01_project_structure.md       # 本文
│   ├── 02_core_contracts.md          # 字段级冻结
│   ├── 03_metric_denominators.md     # coverage / semantic / strict
│   ├── 04_paired_regression_p1a.md   # baseline/candidate compare (P1-A)
│   ├── 05_bootstrap_statistics_p1b.md
│   ├── 06_slices_and_gates_p1c.md
│   └── 07_dimension_contracts.md     # D0–D10 一页纸 Contract（planned）
│
├── configs/                          # 用户/示例 YAML（NN_verb_object）
│   └── examples/
│       ├── 01_score_toy_multiclass.yaml
│       ├── 02_score_toy_metric_swap.yaml
│       ├── 03_score_n2s_offline_replay.yaml
│       └── 04_score_json_extraction.yaml
│
├── src/linguaeval/
│   ├── __init__.py
│   ├── core/
│   │   ├── schema/                   # pydantic/jsonschema contracts
│   │   ├── registry.py
│   │   ├── runner.py                 # offline + online entry
│   │   └── manifest.py
│   ├── adapters/
│   │   ├── dataset/
│   │   ├── model/                    # vllm, openai_compatible, hf
│   │   └── external/                 # lm_eval, inspect（planned）
│   ├── parse/
│   ├── metrics/
│   ├── analysis/                     # slices, badcases, stats
│   ├── packs/
│   │   ├── integrity/                # D0
│   │   ├── business/                 # D1（通过 TaskScorer）
│   │   ├── schema_if/                # D2
│   │   ├── regression/               # D3
│   │   ├── language/                 # D4 registry + ind stub…
│   │   ├── robustness/               # D5
│   │   ├── context/                  # D6
│   │   ├── calibration/              # D7
│   │   ├── consistency/              # D8
│   │   ├── safety/                   # D9
│   │   └── efficiency/               # D10
│   └── reports/
│       └── markdown.py
│
├── examples/
│   ├── indonesian_n2s/               # Example-01：复杂业务 reference
│   ├── toy_multiclass/               # Example-02：证明非 N2S 绑定
│   └── toy_extraction/               # Example-03：结构化抽取
│
├── data/
│   ├── smoke/                        # ≤64 条级 smoke
│   └── fixtures/                     # 离线 prediction fixtures
│
├── results/                          # NN_verb_object run artifacts
├── logs/
├── scripts/                          # verb_object English
└── tests/                            # unit + contract + smoke gates
```

### 结果产物约定（每个 run）

```text
results/01_eval_<task>_<model_tag>/
├── manifest.json
├── data_audit.json              # D0（若启用）
├── predictions.jsonl            # PredictionRecord 流
├── business_metrics.json        # D1
├── schema_metrics.json          # D2
├── retention_matrix.json        # D3（有 base+sft 时）
├── slices.json                  # 固定 slice
├── badcases.jsonl
├── gate.json
├── report.md
└── (later) robustness_*.json / calibration_*.json / ...
```

命名：`results/NN_verb_object/`，禁止 `results/run_20260810/` 作主键。

---

## 6. Examples Strategy

| Example | 作用 | 何时 |
|---------|------|------|
| `indonesian_n2s` | 证明 Kernel 能承载复杂企业任务（multi-target + schema + dialogue + imbalance） | P0 fixture / P1 full |
| `toy_multiclass` | 证明 Kernel **不依赖** N2S/JSON 多字段 | **P0 验收硬门禁** |
| `toy_extraction` | 证明 field-level schema scorer | P1 |

现有资产映射（只读来源，迁移时适配，不反向污染 Kernel）：

```text
LlamaFactory/tests/yewupingce/n2s_test/
├── n2s_result/*.json            → PredictionRecord fixtures
├── n2s_result/*_metrics.json    → 重算一致性金标
├── data/dialogue_*.json         → DatasetAdapter 输入
└── REPORT_base_vs_sft.md        → report 叙事参考
```

---

## 7. Phase Roadmap（收敛版）

### P0 — Generic Kernel Contract + Offline Scoring

**做：**

- 冻结 7 Contracts（本文结构 + 下一篇字段级）  
- `DatasetAdapter`（含 N2S prediction→PredictionRecord）  
- `Parser` + 最小 `TaskScorer`（binary / multiclass / joint stub）  
- `Aggregator` + `RunManifest`  
- Pack registry 骨架；D0–D10 Contract 一页纸（多为 `planned`）  

**不做：** 完整推理、十维实现、HTML、Safety、自研 MMLU。

**验收（必须全过）：**

1. **A. N2S offline replay**：现有 prediction 不重新推理，重算主指标（至少 P/R/F1、format_rate）与现报告一致（允许文档化的舍入误差）。  
2. **B. Toy multiclass**：32–64 条文本分类走同一 Kernel，产出 accuracy / macro_f1。  
3. **C. Metric swap**：同一 `predictions.jsonl` 仅改 MetricSpec（如 F1→F2）可重算，无需重跑模型。

### P0.5 — Harden Kernel（可信 / 可比较）

| Slice | Goal |
|-------|------|
| **A** | DatasetAdapter registry；primary metric 配置化；清 Kernel 业务泄漏 |
| **B** | Parser/Validator；`from_raw` / `from_parsed`；ScoreRecord |
| **C** | fingerprint / provenance；`data_audit.json`；coverage + semantic/strict 双口径（见 `docs/03_metric_denominators.md`） |

**不做（P0.5）：** IndoMMLU、Calibration、Bootstrap、在线推理。

### P1 — Paired Regression + Statistics（D3）

| Slice | Goal |
|-------|------|
| **A** | `baseline`/`candidate` align + ComparisonRecord + 4-cell + mini metric Δ + cases（见 `docs/04_paired_regression_p1a.md`） |
| **B** | paired / cluster bootstrap CI（见 `docs/05_bootstrap_statistics_p1b.md`） |
| **C** | fixed slices + CI-aware gate（见 `docs/06_slices_and_gates_p1c.md`） |

**P1 不做：** IndoMMLU 接入、鲁棒扰动、自动 slice discovery。

### P1 (later packs) — Schema + telemetry leftovers

- D0 Integrity（overlap / missing gold / duplicate / class & language coverage）  
- D1 multi-target + joint as comparable derived target  
- D2 schema/IF 细分指标  
- D10 基础 timing/usage 写入 PredictionRecord  

### P1.5 — Calibration

- scores API → threshold sweep / PR / ECE / Brier / Risk-Coverage  
- `decision_curve.json`

### P2 — Reliability trio

- D5 PerturbationSpec（首批 5 类印尼向扰动，接口通用）  
- D6 Context strategy（optional conversation）  
- D8 Consistency（共享 paired-eval 基建）

### P3 — Multilingual

- LanguagePack registry（接口通用；`ind` 先落地）  
- Belebele + 1 knowledge/culture；lm-eval adapter 可选  
- 第二语言（如 `ara`）证明非印尼硬编码  

### P4 — Safety + Production

- D9 SafetyPack  
- 完整 D10 protocol（warmup/concurrency/comparable flag）  
- GateEngine 泛化 + CI smoke  
- HTML 最后再说  

---

## 8. Gate Engine（形状，不写死 N2S）

Gate 只理解：

```text
metric_path  +  operator  +  threshold
```

示例（用户可换）：

```yaml
gates:
  business.targets.n2s.recall: { min: 0.70 }
  schema.json_valid: { min: 0.995 }
  regression.indommlu: { max_drop: 0.03 }      # 有跑才启用
  efficiency.p95_latency_ms: { max: 500 }      # comparable=true 才启用
```

产出 `gate.json`：`PASS | FAIL` + 逐条明细。

---

## 9. README 披露规则

README 必须分两表：

```text
## Supported
（通过 No Empty Pack Rule 的 Pack）

## Planned
（仅有 Contract / 部分实现）
```

禁止在只有目录占位时写「支持十维评测」。

---

## 10. Explicit Non-Goals（现阶段）

- 自研整套 MMLU / 五十语言一次性接入  
- 用 LLM Judge 给分类/JSON 打分  
- 单一总分替代多维 Model Card  
- 把 Inspect/lm-eval 做成 Core  
- 空 Pack 目录刷存在感  
- 继续把 Kernel API 设计成 N2S 字段名  

---

## 11. Immediate Next Documents / Tasks

| Order | Deliverable | Purpose |
|------:|-------------|---------|
| 0 | `docs/00_executive_brief.md` | 向上汇报口径（已入库） |
| 1 | `docs/01_project_structure.md` | 结构安排（本文） |
| 2 | `docs/02_core_contracts.md` | Contract 字段级冻结（含 ScoreRecord / provenance） |
| 3 | `docs/03_metric_denominators.md` | coverage / semantic / strict（P0.5-C） |
| 4 | `docs/04_paired_regression_p1a.md` | baseline/candidate paired kernel（P1-A） |
| 5 | `docs/05_bootstrap_statistics_p1b.md` | paired / cluster bootstrap CI（P1-B） |
| 6 | `docs/06_slices_and_gates_p1c.md` | fixed slices + CI-aware gate（P1-C） |
| 7 | Offline kernel + compare configs `05_`/`06_` | P0.5 + P1 验收 |
| 8 | P1.5 Calibration | 下一步主线之一 |
| 9 | `docs/07_dimension_contracts.md` | D0–D10 一页纸 Contract（planned） |

### Naming habit（硬性）

```text
results/NN_verb_object/
docs/NN_descriptive_english.md
configs/examples/NN_verb_object.yaml
```

禁止：`phase*` 目录名、双 `00_` 撞号、configs 无编号。

**Note：** P0 离线内核与 Example adapters 已开始落地；空 Pack 实现目录仍禁止伪装为 supported。

---

## 12. Decision Log（本轮已拍板）

1. 品牌 LinguaEval；仓库目录 `eval_factory`。  
2. 十维保留为统一设计面；实现分阶段。  
3. D3 = `ModelComparator`，不是独立 benchmark。  
4. Contracts ≥ 7（含 OutputSpec / ModelSpec / PredictionRecord / RunManifest）。  
5. P0 以 **offline scoring kernel** 为主，双/三验收（N2S replay + toy multiclass + metric swap）。  
6. Inspect / lm-eval = optional adapters。  
7. Report = Markdown first。  
8. P1-A：Kernel 角色名 `baseline`/`candidate`；单 `compare.target`；semantic 默认；sample_id 严格对齐；applicable 排除四格；产物编号 `05_`/`06_`。  
9. N2S P1-A reference：`content_Indonesian_multi_skill_qwen3_4b_base_en.json` vs `qwen3_4b_test3.json`（同评测集；不完整 base run 会被严格对齐拒绝）。  
