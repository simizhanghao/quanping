# LinguaEval — 向上汇报简版（Stakeholder Brief）

**Status:** 立项口径冻结  
**Date:** 2026-08-10  
**配套结构文档:** `docs/01_project_structure.md`

---

## 一、为什么要做这个项目

我们现在已经训练出了一个业务 SFT 模型，但真正到了“评模型”这一步，会发现只看一个 F1 或 Recall 是不够的。

例如一个模型业务 F1 提升了，可能同时发生：

- 通用能力退化；
- JSON 格式变差；
- 小语种口语和 ASR 噪声下不稳定；
- 换一种表达预测就翻转；
- 模型置信度很高但其实预测错了；
- 延迟和显存成本不适合上线。

所以这个项目解决的核心问题不是：

> “怎么跑几个 benchmark？”

而是：

> 一个业务模型 SFT 完成以后，怎样系统回答：它到底有没有变好、哪里变好了、牺牲了什么、是否稳定、是否适合上线？

因此最终目标是做一个 **面向小语种业务 SFT 模型的通用回归评测系统 LinguaEval**。

---

## 二、和普通评测框架最大的区别

普通做法通常是：

```text
模型 → MMLU / C-Eval / 某业务测试集 → 几个分数
```

我们想做的是：

```text
             Base Model
                 │
             SFT Model
                 │
                 ↓
           LinguaEval
                 │
 ┌───────────────┼────────────────┐
 ↓               ↓                ↓
业务学会了吗？ 通用能力掉了吗？ 生产环境稳定吗？
 ↓               ↓                ↓
业务指标       Base→SFT回归      鲁棒/校准/效率
                 │
                 ↓
             Model Card
                 │
                 ↓
          是否可以 Release
```

也就是说，评测对象不是孤立的 SFT 模型，而是 **Base → SFT 发生了什么变化**。

例如：

```text
业务 Recall +10%，但 IndoMMLU -8%、JSON 成功率 -3%、ASR 噪声下 Flip Rate 30%。
```

这种模型显然不能简单说“训练成功”。

---

## 三、核心设计理念：通用 Kernel + 可插拔 Evaluation Pack

项目不是写成一个 N2S 专用脚本。

N2S 只是我们第一个真实案例，因为它非常复杂，刚好包含：

- 二分类 N2S；
- 多分类 routing；
- conditional intent；
- JSON 格式；
- 印尼语；
- 多轮上下文；
- Base/SFT 对比；
- latency；
- calibration。

因此我们用它来验证框架，但底层不能出现：

- n2s 专用字段（作为 Kernel 唯一路径）
- banking 专用逻辑
- routing 专用代码

底层只认识通用流水线：

```text
Dataset → SampleRecord → TaskSpec → PromptBuilder → ModelAdapter
       → PredictionRecord → Parser → Scorer → Aggregator → Report
```

例如 N2S 可以定义三个 target：

```text
decision → binary
routing  → multiclass
intent   → conditional text
```

别人拿去做风控，可以换成：

```text
fraud      → binary
risk_level → multiclass
reason     → text
```

Kernel 不需要改。**Metric 和 Task Type 绑定，而不是和具体业务绑定。**

---

## 四、设计十个评测维度（四层 + D3 横切）

**第一层：评测可信性**  
D0 Data Integrity：训练/测试泄漏、缺失标签、重复样本、有效样本数。  
数据口径不可信，后面所有分数都没有意义。

**第二层：模型到底学会了什么**  
D1 Business；D2 Schema/IF；D4 Language；D9 Safety（按需）。

**第三层：模型是否可靠**  
D5 Robustness；D6 Context；D7 Calibration；D8 Consistency。

**第四层：能不能上线**  
D10 Efficiency（TTFT、Latency、Throughput、VRAM、Failure Rate）。

**D3 Base→SFT Regression 贯穿所有层**——本质不是一个 benchmark，而是：

> 模型前后能力变化分析器 `Compare(model_A, model_B, metric)`。

---

## 五、开发阶段

```text
P0  Generic Evaluation Kernel（协议 + 离线打分）
P1  Business + Schema + Base/SFT Regression（N2S Reference Case）
P1.5 Calibration
P2  Reliability（Robustness + Context + Consistency）
P3  Multilingual Language Pack
P4  Production（Efficiency + Release Gate + CI）
```

不是“重新造一个 HELM”。公开 benchmark 可通过 lm-eval adapter 接入；Inspect 可作为插件。  
真正自己做的是企业 SFT 最缺的部分：**业务回归、Schema、对话、鲁棒性、校准和 Release Gate。**

---

## 六、使用方法（目标形态）

用户只提供：

```yaml
base_model: xxx
sft_model: xxx
language: ind
business:
  dataset: xxx.json
  task_spec: xxx.yaml
  output_schema: xxx.yaml
packs: [business, schema, retention, robustness, calibration]
```

运行：

```bash
linguaeval run config.yaml
```

得到 `data_audit.json`、`predictions.jsonl`、各类 metrics、`badcases.jsonl`、`gate.json`、`report.md`。

最终不是简单告诉用户 `F1=0.82`，而是告诉：

- 业务能力提升多少；
- Base→SFT 哪些能力退化；
- 哪类样本最容易错；
- 小语种噪声下是否稳定；
- 模型什么时候不可信；
- latency/VRAM 是否适合部署；
- 最终 Release PASS 还是 FAIL。

---

## 七、与结构文档的关系

| 文档 | 用途 |
|------|------|
| 本文 `00_executive_brief.md` | 立项 / 向上汇报口径 |
| `01_project_structure.md` | 工程结构、Contract、目录、验收 |
| `02_core_contracts.md` | 字段级协议（随代码冻结） |
| `03_dimension_contracts.md` | D0–D10 一页纸（planned） |
