# Pi 噪声与概念瓶颈模型

本项目研究 Pi 噪声在视觉概念瓶颈模型（CBM）中的机制与效用。各基线模型以 Git submodule 形式独立维护，主仓库记录可复现的代码版本与实验报告。

## 子仓库概览

| 目录 | 基线模型 | 远端仓库 | 主要报告 |
| --- | --- | --- | --- |
| [LF-Pi](LF-Pi/README.md) | LF-CBM | [Dodocheshire/LF-Pi](https://github.com/Dodocheshire/LF-Pi) | [NOISE_MECHANISM_REPORT.md](LF-Pi/NOISE_MECHANISM_REPORT.md) |
| [D-Pi](D-Pi/PI_README.md) | DCBM (ICML 2025) | [Dodocheshire/D-Pi](https://github.com/Dodocheshire/D-Pi) | [EXPERIMENT_REPORT.md](D-Pi/EXPERIMENT_REPORT.md) |
| [PS-Shared-Pi](PS-Shared-Pi/README.md) | Partially Shared CBM (AAAI 2026) | [Dodocheshire/PS-Shared-Pi](https://github.com/Dodocheshire/PS-Shared-Pi) | [EXPERIMENT_RESULT.md](PS-Shared-Pi/EXPERIMENT_RESULT.md) |
| [SALF-Pi](SALF-Pi/EXPERIMENT_REPORT.md) | SALF-CBM (CVPR 2025) | [Dodocheshire/SALF-Pi](https://github.com/Dodocheshire/SALF-Pi) | [EXPERIMENT_REPORT.md](SALF-Pi/EXPERIMENT_REPORT.md) |
| [PiNI](PiNI/experiment_report.md) | 空间加噪 CBM | [Dodocheshire/PiNI](https://github.com/Dodocheshire/PiNI) | [experiment_report.md](PiNI/experiment_report.md) |

## 快速开始

```bash
git clone --recurse-submodules https://github.com/Dodocheshire/Pi-CBM.git
```

子模块管理、环境依赖及更新流程详见 [SUBMODULES.md](SUBMODULES.md)，版本映射详见 [submodules.json](submodules.json)。

## 重点实验与设计文档

- **PS-Shared 基线与消融实验**：详见 [PS-Shared-Pi/EXPERIMENT_RESULT.md](PS-Shared-Pi/EXPERIMENT_RESULT.md)。
- **LF-Pi 噪声机制与可视化**：详见 [LF-Pi/NOISE_MECHANISM_REPORT.md](LF-Pi/NOISE_MECHANISM_REPORT.md)。
- **PiNI 空间加噪实验报告**：详见 [PiNI/experiment_report.md](PiNI/experiment_report.md)。
- **共享扰动视图方案设计**：详见 [nextstep.md](nextstep.md)。
