# Dual-Mask-Full v0.1 实现说明

**arm**：`S0_DUALMASK_FULL_V01`　**基于**：`ff5ad1d4b918d56c6bfa48a2870dc5223e757237`（已验证的 clean 双MASK masked@500 实现）

## 1. 唯一的目标函数改动

```
旧：L_old = 10*L_S + 2*S_S + 1*L_U
新：L_full = 10*L_S + 2*S_S + 10*L_U + 2*S_U
```

四个系数从第 1 步起固定，不做 warmup／动态调权／梯度平衡，不在总 loss 外再乘任何因子：

| 项 | 定义 | 权重 |
| --- | --- | --- |
| `L_S` | 原 S0 的 SIDM + DISM（`compute_smartclip_terms`，未改动） | 10（内部常量，仍由 helper 施加） |
| `S_S` | 原 S0 的 `mean(abs(m_S))` | 2（同上） |
| `L_U` | 后缀双向 CE（`suffix["loss"]`，含一次 `W/V`） | `--lambda-suffix`，本轮 10.0 |
| `S_U` | **仅有效正配**的 U 门 `mean_d(abs(m_U))` 全局均值 | `--lambda-u-sparse`，本轮 2.0 |

`S_U` 的定义（全局）：`S_U = (1/V) * Σ_{i: valid_i} mean_d(abs(m_U[i, i, d]))`，
`mean_d` 只除一次特征宽度 `D=512`；不使用全 `B×G` 门均值，不计无效后缀样本。

## 2. 代码改动（最小）

* `model/dual_mask_suffix.py`
  * 新增常量 `U_SPARSE_LAMBDA = 0.0`；模块新增参数 `lambda_u_sparse`（native 模式下非 0 直接报错）。
  * `pairwise_masked_scores(..., rank=None, valid_local=None)`：在**同一分块评分**的同一份 `m_U` 上抽取正配门。
    局部行 `a` 的全局正配列是 `rank*B + a`，因此每个有效正配只在包含该列的那一个瓦片里被取一次、
    且绝不用瓦片对角线冒充全局正配；只用整数瓦片代数定位，不产生 GPU 同步，也不保存 `[B,G,D]` 张量。
    同时记录正配门的 keep ratio、`p` 均值、全开/全关比例、`cos(normalize(g*m_U), normalize(g))`（均为 detached）。
  * 新增 `u_sparse_terms(...)`：把局部稀疏和归约为
    * 反传标量 `(W / V) * Σ_local`（DDP 对 W 个 rank 求梯度均值后恰好等于 `(1/V) * Σ_global`）；
    * 日志标量 `S_U_global = all_reduce(Σ_local) / V`（日志归约不参与训练，所有 rank 同序参与）。
  * 日志字段：未加权的 `L_S / S_S / L_U_global / S_U_global`，加权的
    `weighted_s0_align / weighted_s0_sparse / weighted_u_align / weighted_u_sparse`，
    本地实际反传值 `loss_total_local_backward`、`loss_u_sparse_backward`（未加权）与
    `weighted_u_sparse_backward`，以及正配门统计与 `u_sparse_positive_count_local`。
* `train/train_dual_mask_suffix.py`
  * 新增 `--lambda-suffix`（默认 1.0）与 `--lambda-u-sparse`（默认 0.0）；默认值即旧版行为，逐位等价。
  * 两个参数**真正进入模型构造与总 loss**；config 中旧的硬编码 `suffix_lambda` 改为参数值，
    并写入 `u_sparsity_lambda` 与 `u_sparsity`（同一数值）与 `total_objective` 字符串。
* 未改动：S0 helper、门结构 `Linear(1024,512)→GELU→Linear(512,512)`、门初始化
  （Xavier、末层 weight=0、bias=log 8）、RNG 隔离、数据管线、`W/V` 归约、通信实现。

## 3. 增量测试（`tests/test_dual_mask_suffix_full_v01.py`，10 项）

1. 新权重确实进入总 loss，且 S0 部分与旧版逐位相同（`10L_S+2S_S` 不变），`L_U` 不随稀疏权重变化；
2. 初始化全开时 `S_U = 1`、加权值为 2，且新旧总 loss 之差恰为 `9L_U + 2`；
3. 正配稀疏索引：非方形瓦片、跨瓦片尾部、非连续 valid 下与**未分块参考**一致，
   计数等于有效数，改动非正配列（无效行）不改变 `S_U`；
4. 稀疏梯度与独立 `abs(hard-ST(sigmoid(logits)))` 参考一致；只到新门（g/t 特征无梯度），
   全关门时 `S_U = 0` 且门参数无梯度（abs 在 0 处次梯度为 0，不加自定义梯度）；
5. 反传值 `(W/V)·Σ_local`、日志值 `Σ_global/V` 的归约语义；
6. 两个新系数可存/读回，完整 F 严格加载，裸学生仍只导出 CLIP state。

旧配置回归由未改动的 `tests/test_dual_mask_suffix.py` 覆盖（含两进程 DDP 总目标对照，共同 20 项全通过）。
两进程 DDP 对照已扩展到新目标：使用小模型与非全开门，覆盖 valid 1 vs 3、零有效 rank、`V < 2`，
逐参数比较梯度与一步更新，并校验 `S_U` 的全局值与本地反传值。
