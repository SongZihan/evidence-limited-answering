# 第一轮关系使用机制实验

本项目执行预先锁定的 Qwen3-8B 公司场景实验：384 条自然生成、三位置逐层读出、完整候选续写概率，以及门槛通过后的有限子空间干预。

## 文件入口

- `protocol.json`：运行前门槛、划分、候选配置和成功标准。
- `data/jobs.json`：全部正文、提示词、条件及金标准。
- `data/lock.json`：材料与协议 SHA256。
- `run.py`：先探索、冻结配置，再验证；禁止覆盖已存在的自然输出。
- `report.py`：只读取已完成的结果，不参与材料或干预选择。
- `results/report.md`、`results/report.html`：最终报告。

## 服务器执行环境

项目目录为 `./jspace_relation_mechanism_round1`，Python 为 `python`。复用 `models/Qwen3-8B` 和既有 `jspace_task_selection` 项目中的官方 J-lens 与权重。单张 GPU、单条推理、非思考、BF16、eager attention、贪心生成。

`launch.py` 在 GPU 1 启动独立后台进程，输出位于 `logs/run.log`，状态位于 `results/status.json`。`transfer.py` 从进程环境读取 SSH 密码，不写入文件。重新运行必须使用新的结果目录或明确归档现有结果，不覆盖本轮留档。

## 解释约束

自然 PN 与强制 P1 分开报告；干预比较使用自然前缀的当前重放作为基线。J-lens 的城市词汇读数不等于关系知识。子空间用所有探索集分句 B−D 配对差构造，不按是否误填选择样本，不按验证集重新选择层或秩。

随机对照仅一个固定种子，匹配浮点状态中的扰动范数；实际写入模型时统一转为 BF16。供体与接收材料的事实及证据条件一致。成功也只支持有限因果作用，不证明已识别独立的关系绑定方向或可部署的纠错方法。