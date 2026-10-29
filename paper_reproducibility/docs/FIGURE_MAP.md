# Current figure-to-data mapping

| Figure | Scientific content | Inputs and scripts |
|---|---|---|
| 1 | Same-text instruction example | Editable `en_img/figure1_instruction_example_en.svg`; original company validation records, template T02/e2 |
| 2 | Company behavior and template differences | `jspace_relation_binding/results/validation.jsonl`; `plot_section32.py` |
| 3 | Development-set layerwise readouts | `jspace_relation_mechanism_round1/data/jobs.json`, `results/readouts.jsonl`; `plot_section34.py` |
| 4 | 13-model company comparison | `section41_data.py` lists all row-level sources; `plot_section41.py` |
| 5 | Ten-scenario comparison | `section42_data.py` maps model-specific four/five-scenario files; `plot_section42.py` |
| 6 | Original independent internal validation | `section43_data.py`; natural outputs, readouts, candidate scores and coverage from `jspace_relation_mechanism_round1`; `plot_section43.py` |
| 7 | Selective intervention | `section44_data.py` for original model comparison, `jspace_core_expanded/report.py` for expanded Qwen results; `plot_fig8_expanded.py` |

分析和绘图脚本保存在`formal/_draft_work/`，仅为兼容相对路径而保留这个目录名；包内不包含手稿编辑脚本。统一调用见根目录`reproduce.py`。图7排除GLM展示，但相应模型的文本记录仍用于完整保留跨模型实验边界。

图2—7保留数据生成脚本，`tools/layout_figures.py`执行第二轮英文排版。图1为定性示例，其SVG是布局源，不通过随机生成重建。当前英文输出均不含内部大标题或说明性脚注；必要解释见论文图注。

## Dependency-only material

`jspace_cross_scenario`提供后续无提示版本复用的数据生成与评分组件；论文跨场景主结果取自`jspace_cross_nohint`等后续结果。`jspace_task_selection`仅保留lens元数据等依赖，不包含早期注意力头扫描。历史修复及输出规范化代码用于追溯最终文本记录，不替代原始评分定义。

`jspace_relation_recovery/data/prior_environment.json`及`jspace_multifield_interference/results/validation.jsonl`仅作为历史环境和动机审计依赖保留，不计入论文新增独立验证。
