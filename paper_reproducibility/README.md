# 论文代码与复现数据 / Paper reproduction package

本包对应《证据不足时大语言模型的作答决策：指令影响与低维干预》。保留论文实际使用的受控材料、文本输出、评分与分析代码；文件夹名称沿用实验阶段名称，以保留跨目录依赖。

## 无需GPU的复现

建议先解压一份工作副本。Python 3.11或以上：

```bash
python -m pip install -r requirements-cpu.txt
python reproduce.py verify
python reproduce.py summarize
python reproduce.py plots
```

`summarize`从逐条结果重新计算跨模型、跨场景、原内部验证、原干预与扩充验证汇总，并与`expected/`中的论文数值比较。`plots`输出到`reproduced/`；`en_img/`保存当前投稿图片。图1为可直接编辑的SVG示例图，其英文内容是中文实验的翻译，并非英文实验。图2—7从数据绘制后统一调整布局。原始绘图脚本文件名保留旧编号，统一入口负责映射到当前图1—7。

首次核验应在重新生成前运行：分析脚本会更新本包内的派生汇总，因此重新生成后文件校验值可能改变。原始逐条数据不修改。

## GPU实验

先阅读`docs/ENVIRONMENTS.md`，分别建立匹配实验阶段的环境。按`configs/paths.example.json`提供本地模型目录，运行：

```bash
python tools/link_models.py --config configs/paths.local.json
python jspace_core_expanded/run.py qwen8b
# 另一个环境中：python jspace_core_expanded/run.py qwen2b
```

GPU脚本应在包根目录执行，建议另建工作副本；它们会写入对应实验的`results/`。本次打包未重新执行GPU实验。不要把CPU核验通过理解为所有模型环境均已重新测试。

内部读出需要自行安装官方Jacobian Lens，并提供原研究使用的lens权重：`jspace_task_selection/results/jacobian_lens.pt`。其校验信息在同目录`lens_metadata.json`。本包不含模型权重和lens权重；缺少原权重时不能声称逐位复现原内部读数。冻结低维基和随机基已包含，不需要重新选择层或重新拟合子空间。

## 数据边界

- 原公司行为验证：6模板，联合条件每格36例。
- 原机制验证：4模板，每格16例；图6继续采用这批内部读数。
- 扩充独立验证：24个新模板，每格96例；图7两个Qwen模型使用这一批结果。
- 跨场景及历史迁移单独保存，不与独立验证合并。
- 自然生成按严格协议评分；干预保留内容评分与严格评分。JSON `null`与字符串`"null"`不同。未定位、截断和格式失败不作为已修复样本。

`docs/FIGURE_MAP.md`列出图表来源，`docs/ENVIRONMENTS.md`记录分阶段依赖。`docs/source_manifest.json`保留源文件散列；`MANIFEST.sha256`对应脱敏后的发布文件。环境路径改写及注释清理会改变代码散列，不改变历史模型检查点与冻结基的散列。旧记录中的历史校验值是原运行溯源信息，不是发布代码的校验值。

## English quick start

Install `requirements-cpu.txt`, then run `python reproduce.py all` in a fresh extracted copy. The CPU route recomputes numerical summaries from saved row-level records and renders seven figures. Publication-layout SVGs and PNGs are also included in `en_img/`.

GPU reruns require separate, stage-matched environments and user-provided model directories. See `docs/ENVIRONMENTS.md`; no GPU reruns were performed during packaging. Model weights, large activation dumps, operational logs and manuscript-editing utilities are excluded. Model-specific native modes and scoring conventions remain distinct. The original and expanded validation sets must not be pooled as a single preregistered validation.

See `THIRD_PARTY.md` for Jacobian Lens attribution and licensing. No new license is assigned to the authors' research code by this packaging operation.
