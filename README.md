# Evidence-Limited Answering

Code and reproducibility data for **Answering Decisions of Large Language Models under Insufficient Evidence: Instruction Effects and Low-Dimensional Interventions**.

This study examines how additional extraction instructions affect the use of textual evidence, how relational phrasing moderates that effect, and whether low-dimensional state replacement can reduce unsupported answers while preserving supported answers. The package includes ten-scenario behavioral evaluations, internal readouts, intervention experiments, and expanded independent validation.

## Quick start

Clone this repository or download its ZIP, then run:

```bash
python restore_data.py
cd paper_reproducibility
python -m pip install -r requirements-cpu.txt
python reproduce.py verify
python reproduce.py summarize
python reproduce.py plots
```

`restore_data.py` uses only the Python standard library. It joins the checked-in `data_parts/`, verifies every part and the complete archive with SHA-256, and restores the full package. It refuses to overwrite modified local files. The partitioned archive is approximately 37.2 MiB; no external data service or Git LFS is required.

Python sources, protocols and documentation are directly browsable under [`paper_reproducibility/`](paper_reproducibility/). Larger records, figure assets and frozen subspaces are restored from the archive. See the [package README](paper_reproducibility/README.md), [figure-to-data map](paper_reproducibility/docs/FIGURE_MAP.md), and [recorded environments](paper_reproducibility/docs/ENVIRONMENTS.md).

## Included and excluded

- Included: synthetic materials, prompts, gold labels, splits, saved textual outputs, scoring and analysis code, figure-generation scripts, frozen low-rank/random bases, and numerical verification fixtures.
- Excluded: model weights, Jacobian Lens weights, large activation dumps, credentials, private server paths, operational logs, and manuscript files.
- Original and expanded validation remain separate. Figure 6 uses the original internal validation; the two Qwen models in Figure 7 use expanded validation.
- CPU numerical reproduction and all seven English figures were checked from an extracted package. GPU experiments were **not** rerun during packaging; use stage-specific environments and user-supplied checkpoints.

## 中文说明

本仓库对应“证据不足时大语言模型的作答决策：指令影响与低维干预”。先运行 `python restore_data.py` 还原完整复现目录，再按包内说明复算指标和绘图。分块文件是完整数据包的一部分，下载时请保留全部文件。构造材料、模型输出及不同阶段的评分口径均予保留，历史迁移结果不与独立验证混合统计。

Third-party code and models retain their own terms. See [THIRD_PARTY.md](paper_reproducibility/THIRD_PARTY.md). No new license is assigned to the research code by this repository upload.
