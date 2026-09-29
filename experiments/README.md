# SAID 实验索引

| 实验 | 目标 | 固定训练／续训 SHA | 状态 |
|---|---|---|---|
| [S0-DualMask-Clean v0.1](s0_dualmask_masked_3epoch/README.md) | `L_S0 + 1 L_U`，新 U 稀疏 0 | `11af80b344c623b27b93069f9be526970c9c950c` | 完成 3651 次更新 / 3 epoch |
| [S0-DualMask-Full v0.1](s0_dualmask_full_v01/README.md) | `L_S0 + 10 L_U + 2 S_U` | `873b43a5bc000528311e22aac01e92045ac898e0` | 完成 3651 次更新 / 3 epoch；另评 step2000 |
| [S0-DualMask 3/0 @500](s0_dualmask_suffix3_500/README.md) | `L_S0 + 3 L_U`，新 U 稀疏 0 | `dcd33877f1f77a901d292190834f1ffd83a049a8` | 完成500次更新及六项检索；与同预算基线对照 |
| [S0-DualMask 3/0 · 3 epoch](s0_dualmask_suffix3_3epoch/README.md) | `L_S0 + 3 L_U`，新 U 稀疏 0 | `873b43a5bc000528311e22aac01e92045ac898e0` | 500→3651；六项评测及本机Clean/Full同预算对照完成 |

[完整3 epoch及历史得分](RESULTS.md) · [机器可读结果](results.json) · [权重与数据](ASSETS.md)

main 中的训练入口保持 Full 分支的实现；默认对齐/稀疏系数为 1/0。历史轨迹与严格续训使用上表和各 REPRODUCE.md 所列固定 SHA。保存旧实现的方式是 Git 提交与实验分支，不用文件夹名称猜测训练版本。
