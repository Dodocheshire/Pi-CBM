# 子仓库管理指南

主项目 `Pi-CBM` 将各基线模型作为 Git submodule 独立管理，主仓库记录固定的子仓库 commit 哈希以确保实验可复现。

## 子模块结构与上游源

| 路径 | 基线模型 | 仓库地址 | 上游来源 (Upstream) |
| --- | --- | --- | --- |
| `LF-Pi` | LF-CBM | https://github.com/Dodocheshire/LF-Pi | 继承原 Pi-CBM 迁移历史 |
| `D-Pi` | DCBM (ICML 2025) | https://github.com/Dodocheshire/D-Pi | 官方开源仓库 (`upstream`) |
| `PS-Shared-Pi` | Partially Shared CBM (AAAI 2026) | https://github.com/Dodocheshire/PS-Shared-Pi | 官方开源仓库 (`upstream`) |
| `SALF-Pi` | SALF-CBM (CVPR 2025) | https://github.com/Dodocheshire/SALF-Pi | `itaybenou/show-and-tell` |
| `PiNI` | PiNI 空间加噪 CBM (AAAI 2025) | https://github.com/Dodocheshire/PiNI | `hyzhang98/PiNI` |

## 常用操作

### 首次克隆与初始化

```bash
git clone --recurse-submodules https://github.com/Dodocheshire/Pi-CBM.git
```

### 已有仓库同步与更新

```bash
git pull --ff-only
git submodule update --init --recursive
```

### 修改与提交子仓库

修改子模块代码后，需先在子仓库内提交并推送，再在主仓库提交更新后的 submodule 指针：

```bash
# 1. 在子仓库中提交并推送
cd PS-Shared-Pi
git add .
git commit -m "Update PS-CBM experiment revision"
git push origin main

# 2. 回到主仓库更新指针
cd ..
git add PS-Shared-Pi
git commit -m "Update PS-Shared-Pi submodule pointer"
git push origin main
```

## 数据与缓存规范

各子仓库中的 `artifacts/`、`outputs/`、模型权重及大型数据集缓存默认不纳入版本控制（仅保留本地），配置中指定具体路径。
