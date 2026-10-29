# Recorded environments and execution limits

| Stage / model family | Recorded Transformers | Recorded Torch |
|---|---:|---:|
| Qwen3-8B behavior, readouts, intervention, transfer, expanded validation | 4.54.0 | 2.10.0+cu130 |
| Qwen3.5-2B / 27B intervention and expanded 2B validation | 5.15.0 | 2.10.0+cu130 |
| Gemma, Muse, GLM supplementary intervention | 5.15.0 | 2.10.0+cu130 |
| Kimi supplementary intervention | 4.48.2 | 2.10.0+cu130 |

These are recorded versions, not a claim that one combined environment supports all models. Per-run `environment*.json`, protocol files and configuration files are authoritative for each stage. Some historical stages do not record a complete dependency lock; do not infer missing versions. Additional packages depend on the model's own loader and official instructions. Native chat templates, thinking modes, truncation rules and boundary fallback are retained in their experiment scripts.

CPU reproduction was tested separately during packaging; see `docs/VALIDATION.json`. CairoSVG requires the Cairo system library. If unavailable, install Cairo via the operating system package manager; numerical summarization does not require Cairo. English plots use DejaVu Sans. Font/rendering-library differences can change pixels without changing numerical content.

For GPU reruns use Linux/CUDA environments matching each stage. Copy `configs/paths.example.json` to `configs/paths.local.json`, map each `models/<name>` to your actual local checkpoint directory, and run `tools/link_models.py`. Existing paths are never replaced by that helper. Install the official Jacobian Lens separately for original internal observations, and supply its required weights. Checkpoint hash guards remain active.

运行GPU代码前使用独立工作副本，防止新输出与保存结果混合。不要在发布包上重新运行`prepare.py`后误把重新生成的数据当作历史锁定材料；优先使用已保存的`data/`。若要全新生成材料，应另建实验目录并重新记录数据锁。
