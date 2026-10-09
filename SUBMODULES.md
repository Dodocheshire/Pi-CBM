# 基线子仓库管理

主项目 `Pi-CBM` 将实验实现保存为四个 submodule，均关联到 `Dodocheshire` 账号下的私有仓库。主项目提交记录固定的子仓库提交；克隆需要对这些私有仓库有访问权限。

| 路径 | 基线 | 独立远端 |
| --- | --- | --- |
| `LF-Pi` | LF-CBM 与已有 Pi 实验 | https://github.com/Dodocheshire/LF-Pi |
| `D-Pi` | DCBM，ICML 2025 | https://github.com/Dodocheshire/D-Pi |
| `PS-Shared-Pi` | Partially Shared CBM，AAAI 2026 | https://github.com/Dodocheshire/PS-Shared-Pi |
| `SALF-Pi` | Spatially-Aware and Label-Free CBM，CVPR 2025 | https://github.com/Dodocheshire/SALF-Pi |

LF 子仓库继承迁移前 Pi-CBM 的历史，包括远端 ImageNet 与 Places365 的更新；现有服务器路径迁移保持不变。D 子仓库继承官方 DCBM 提交历史，并通过 `upstream` remote 与 `UPSTREAM.json` 保存来源。PS 子仓库在 `official/` 中保留逐文件核验的官方源码，`upstream` 指向原作者仓库。以后新增实现目录按同样方式独立建仓库，再注册为 submodule；只有论文笔记的 `Essay` 候选目录无需建独立代码仓库。

SALF 子仓库保留官方完整历史，`upstream` 指向 `itaybenou/show-and-tell`；本地新增的空间加噪封装及基线复验见 `SALF-Pi/EXPERIMENT_REPORT.md`。官方完整训练代码尚未发布，当前入口基于已发布预训练模型适配。

首次获取：

```bash
gh auth login
git clone --recurse-submodules https://github.com/Dodocheshire/Pi-CBM.git
```

已有主仓库更新：

```bash
git pull --ff-only
git submodule update --init --recursive
git submodule status
```

修改一个基线后，先在子仓库提交并推送，再在主仓库提交更新的指针。以下以 PS 为例：

```bash
cd PS-Shared-Pi
git add ps_pi configs tests README.md
git commit -m "Describe the validated experiment change"
git push origin main
cd ..
git add PS-Shared-Pi
git commit -m "Update PS-CBM experiment revision"
git push origin main
```

`git submodule update --init --recursive` 获取主项目记录的版本，保持实验代码可复现。需要升级某个基线时，在该子仓库中明确选定和验证新提交，再更新主项目指针。通常不要直接使用 `git submodule update --remote` 批量改变版本。

各子仓库的 `artifacts/`、`outputs/`、训练权重、Python 缓存保留本地；迁移不会移动或删除这些目录。D 官方历史随仓库发布的少量示例特征保留历史与来源。`Essay/` 继续作为本地主项目的论文与报告目录，本次注册不改变其内容。机器特定的数据集路径仍由配置指定，新机器须准备数据与模型缓存。
