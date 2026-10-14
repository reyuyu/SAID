# HNS-v1 ViT-L/14 Full 5-Epoch Evaluation

本实验从共同 Step0 独立训练至 6085 updates，唯一训练预算变化是将 scheduler horizon 从 4868 延长到 6085；没有从历史 Step4868 恢复，也没有执行第六 Epoch。训练验收通过，完成 6085/6085 次同步更新，四 rank 参数同步，未检测到 NaN、Inf、OOM 或 DDP 故障。

评测使用裸学生、原生 ViT-L/14、248-token tokenizer、原始图像预处理、L2 归一化和全候选池内积检索。没有使用推理 mask、融合、rerank、ensemble 或 TTA。评测产物位于服务器 runtime 的 `evaluation_final/step4868` 与 `evaluation_final/step6085`；本报告只提交小型审计和结果摘要。

## 汇总指标

| 节点 | scheduler | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I | Urban Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 历史 Full4 @4868 | 4868 | 77.890862 | 83.191078 | 90.670000 | 71.390 | 95.300 | 95.200 | 95.250 |
| 本次 Full5 @4868 | 6085 | 77.938640 | 82.349067 | 90.700002 | 71.323 | 95.400 | 95.600 | 95.500 |
| 本次 Full5 @6085 | 6085 | 77.968310 | 82.423184 | 90.775002 | 71.286 | 95.300 | 95.700 | 95.500 |

相对历史 Full4，Full5@6085 的 Score5 为 +0.077448 pp，J_long3 为 -0.767894 pp，J_long 为 +0.105002 pp，Short4 为 -0.104 pp；Urban I2T 为 0.0 pp，Urban T2I 为 +0.5 pp。由于 Full5@4868 使用 6085-horizon scheduler，它与历史 Full4@4868 的学习率轨迹不完全相同，因此该节点只作轨迹诊断，不能视为严格同调度复现。

## 五集原始结果

完整 30 项 Recall 位于 `RESULTS_4868.json` 和 `RESULTS_6085.json`。其中 Full5@6085 的主要 R@1 为：COCO I2T/T2I 64.620/48.764，Flickr30k-Test1K 92.700/79.060，DOCCI 85.840/86.260，Long-DCI 重建版 65.022/66.417，Urban-1k 95.300/95.700。Long-DCI 是固定 7602 对重建协议，不等同于 DCI-full。

补充的 Flickr30k-Full 官方 token 评测结果见 `FLICKR_FULL_RESULTS.json` 和 `FLICKR_FULL_EVALUATION.md`。Full5@6085 的 Flickr-Full I2T R@1/5/10 为 64.701255/84.611270/90.375358，T2I R@1/5/10 为 44.409275/67.100022/75.360413；Full5@4868 的对应结果为 64.710694/84.611270/90.353334 与 44.405500/67.085549/75.349086。该 Full 协议与 Test1K 是不同数据集范围，不能互相替代。

## 结论与限制

第五 Epoch 没有带来 J_long3 或 Short4 的改善；它保持了总体 Score5 和 Urban Mean，并使 Urban T2I 相对历史 Full4 上升 0.5 pp，但 Urban I2T 没有同步上升。该单 seed、公开测试集反复观察的结果不足以证明第五 Epoch 优于 Full4，也不足以支持继续第六 Epoch。训练、评测和审计进程已经退出，四张 GPU 已释放。
