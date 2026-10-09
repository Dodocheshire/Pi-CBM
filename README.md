# Pi 噪声与概念瓶颈模型

本项目研究 Pi 噪声在视觉概念瓶颈模型中的作用。不同基线的实现以 submodule 独立管理，远端关联 `Dodocheshire` 的 GitHub 账号，主仓库记录可复现的代码版本。

| 目录 | 基线与内容 | 远端 |
| --- | --- | --- |
| [LF-Pi](LF-Pi/README.md) | LF-CBM 原实验、配置与报告 | [Dodocheshire/LF-Pi](https://github.com/Dodocheshire/LF-Pi) |
| [D-Pi](D-Pi/PI_README.md) | DCBM（ICML 2025）官方实现与 Pi 研究 | [Dodocheshire/D-Pi](https://github.com/Dodocheshire/D-Pi) |
| [PS-Shared-Pi](PS-Shared-Pi/README.md) | Partially Shared CBM（AAAI 2026）最终实验与图表 | [Dodocheshire/PS-Shared-Pi](https://github.com/Dodocheshire/PS-Shared-Pi) |
| [SALF-Pi](SALF-Pi/EXPERIMENT_REPORT.md) | SALF-CBM（CVPR 2025）官方空间模型、预训练基线复验与可微加噪入口 | [Dodocheshire/SALF-Pi](https://github.com/Dodocheshire/SALF-Pi) |
| [PiNI](PiNI/EXPERIMENT_RESULT.md) | PiNI（AAAI 2025）官方复现与概念条件 RGB 空间加噪 CBM | [Dodocheshire/PiNI](https://github.com/Dodocheshire/PiNI) |

```bash
git clone --recurse-submodules https://github.com/Dodocheshire/Pi-CBM.git
```

子仓库为私有仓库，需使用有访问权限的 GitHub 账号。已有工作区初始化、更新与子仓库提交方式见 [SUBMODULES.md](SUBMODULES.md)。大型训练缓存保留本地；版本登记见 [submodules.json](submodules.json)。

本地 `Essay/` 保存论文、研究方案与结果图表；DCBM 精简主要结果位于 `Essay/D/report/`，新阶段结果整理到 `Essay/PS-Shared/report/`。PS-Shared 的新基线指 Partially Shared CBM，与论文笔记 `Essay/PS` 的 Post-hoc Stochastic CBM 不同。

LF 实验从 `LF-Pi/` 启动，DCBM 从 `D-Pi/` 启动，PS 新阶段从 `PS-Shared-Pi/` 启动。实验结论以对应的正式报告和验证记录为准。

PS-Shared最终主要结果见[EXPERIMENT_RESULT.md](PS-Shared-Pi/EXPERIMENT_RESULT.md)：三个数据集五种子干净基线测试、CUB七组控制的独立五种子测试、配对区间、属性保护及概念输出精简。共享噪声未证明额外准确率收益；默认确定性适配模型在测试前选定。

LF-Pi 的背景机制核查与真实加噪可视化见 [NOISE_MECHANISM_REPORT.md](LF-Pi/NOISE_MECHANISM_REPORT.md)：主要增益可由任务投影和确定性均值适配解释，尚未证实背景去干扰或真实概念精确率提升。下一阶段优先选 SALF-CBM，已复验官方 ImageNet 基线为 75.56%；目前仅完成加噪入口验证，没有 SALF 加噪提升结论。

当前新目标转向 PiNI 的真实图像噪声机制，并开发带原生概念瓶颈的空间模型。进度、固定验证样本的真实 μ/σ² 可视化、概念精度与匹配控制见 [PiNI/EXPERIMENT_RESULT.md](PiNI/EXPERIMENT_RESULT.md)。PiNI 原方法不是 CBM；新增实现严格通过 18 个命名概念分类，使用概念/区域监督，推理不读取真值掩膜。
