# GRIM

GRIM（GAT-powered Reverse Influence Maximization with Hybrid Decision Design）是一个面向“影响力最大化（Influence Maximization, IM）”研究的代码与数据仓库。项目核心思路是将 Reverse Influence Sampling（RIS）生成的样本/统计量融入图神经网络（以 GAT 系列为主）的训练与选种流程，并提供 IC/LT/SIS 等扩散模型下的仿真评估工具。

## 目录概览

- [new_grim.py](new_grim.py)：一个端到端示例脚本（读取带 seed/coverage 的 `.SG`，训练 VAE + SpGATv2，并通过优化潜变量生成种子集合，再做扩散评估）。
- [ris.py](ris.py)：RIS/RR-set 相关工具，用于从图邻接矩阵构建 RR sets，并生成带 seed/coverage 的 `.SG` 文件。
- [main/](main/)：
  - [main/utils.py](main/utils.py)：通用工具（稀疏矩阵处理、扩散仿真评估 `diffusion_evaluation(_v2)` 等）。
  - [main/model/](main/model/)：模型实现（VAE、GAT/SpGAT/SpGATv2、GIN 相关组件等）。
