---
name: command-presentation
description: >-
  Presents every shell/git/pytest/python command as a single standalone
  copy-paste bash block with a bold Chinese title. Use whenever the agent
  gives the user commands to run, including push, commit, pytest, smoke,
  install, or any terminal step under user-executes-commands.
---

# Command Presentation (Standalone Blocks)

Hard skill. Complements `01-user-executes-commands`: the **user** runs shell;
the agent only prints commands — and **must** print them in the form below.

## When to Apply

Any time the reply includes a command for the user to execute:
git, pytest, python -m, pip, conda, docker, mkdir, curl, etc.

## Hard Rules

1. **One primary command block per turn** (one verifiable step). Do not scatter
   commands across prose, bullets, or multiple competing fences.
2. **Never** run the Shell tool for these commands; only present them.
3. Commands must be **copy-paste ready**: absolute `cd` when cwd matters, no
   fake placeholders like `<path>` unless explicitly marked optional.
4. After the fence, always tell the user to paste terminal output back.
5. Do **not** bury the block inside a long explanation; title → fence → short
   expectation line.

## Required Output Form

Use this structure every time (title adapts to purpose):

```markdown
**{动作}命令（请你执行）** — {一句话目的}

\`\`\`bash
cd /absolute/workdir

command_1
command_2
\`\`\`

执行后把终端输出贴回。
```

### Title examples

| Situation | Title |
|-----------|--------|
| Commit + push | `**Push 命令（请你执行）** — 提交并推送 P1.5-B` |
| Tests only | `**验收命令（请你执行）** — P1.5-B pytest + smoke` |
| Inspect only | `**检查命令（请你执行）** — git status / diff` |
| Install | `**安装命令（请你执行）** — 安装 dev 依赖` |

Chinese title is preferred in this project. English titles only if the user
writes in English.

## Canonical Examples

### Push / commit

```markdown
**Push 命令（请你执行）** — 提交并推送 P1.5-B

\`\`\`bash
cd /data/hanchengcheng/hcc_1/eval_factory

git status
git diff --stat
git log -3 --oneline

git add \
  README.md \
  src/linguaeval \
  docs \
  configs/examples \
  examples/toy_calibration \
  tests

git status

git commit -m "$(cat <<'EOF'
Add P1.5-B calibration metrics on ConfidenceRecords.

Compute ECE/Brier/NLL/AUROC for AVAILABLE scores; free-gen without scores stays NOT_AVAILABLE.
EOF
)"

git push origin HEAD

git log -1 --oneline
git status
\`\`\`

执行后把终端输出贴回。
```

### Acceptance / pytest

```markdown
**验收命令（请你执行）** — P1.5-B 单测与全量回归

\`\`\`bash
cd /data/hanchengcheng/hcc_1/eval_factory

PYTHONPATH=src python -m pytest -q tests/test_calibration_p15b.py
PYTHONPATH=src python -m pytest -q
\`\`\`

执行后把终端输出贴回。
```

## Anti-Patterns (Forbidden)

```text
❌ 用途: ...
   目录: ...
   命令:
   <cmd>                    ← old template; do not use when this skill applies

❌ Run: `pytest -q` then also `git status` in separate paragraphs

❌ Inline only: 请执行 git push origin HEAD

❌ Multiple ```bash fences in one turn for unrelated steps
```

## Relationship to Other Rules

- Ownership / no Shell tool: `.cursor/rules/01-user-executes-commands.mdc`
- One task at a time: `task-scoped-execution`
- This skill only governs **presentation shape**, not whether the user should
  run the command.
