# SAID-C1-TCR v0.1 — 文本条件视觉特征重建（独立消融）

**C1-TCR**（Text-Conditioned visual feature Reconstruction）：给定当前已观测文本 `C_S` 与互补视觉输入
`r_U`，预测同一张图由**冻结初始化的 CLIP 视觉塔**产生的完整视觉特征。

一句话：用冻结初始视觉塔提供完整图像特征目标，训练"当前未被 Said 选择的视觉输入 ＋ 已观测文本"
去预测它；让预测分支追逐固定参考，而不是让原生 Global 追逐内部合成答案；最后仍以**丢掉 decoder
后的原生学生 CLS 检索**验收。

| 项 | 值 |
| --- | --- |
| worktree | `/root/SAID-c1-tcr` |
| branch | `codex/said-c1-tcr-v01` |
| base SHA | `32925bc62e9cfd8dba571022c9e4cb3d5dcc7d70`（已确认的干净 base，未回退） |
| objective | `said_cls_tcr` |
| arm | `C1_text_conditional_reconstruction` |
| 本轮训练 | **1 次**，500 optimizer steps |
| 结果判定 | **AUXILIARY_TASK_IMPROVED_ONLY**（重建改善，原生检索未改善） |
| 是否跑 3 epoch | **未跑**（按约定 STOP） |

旧 C0 工作目录与分支**未被触碰**（新建独立 worktree，未 merge/rebase/squash/amend/force-push）。

## 1. 冻结的数学定义与默认配置

$$
v=E_{I,\theta}(I_a),\quad g=\mathrm{Norm}(v),\quad
(t_{raw},T)=E_{T,\phi}(C_S),\quad t=\mathrm{Norm}(t_{raw})
$$

$$
p=\sigma(M_\omega(\mathrm{sg}(T))),\quad
m_S=(p\ge0.5).\mathrm{float}-p.\mathrm{detach}+p,\quad
m_U=1-\mathrm{sg}(m_S)
$$

$$
\boxed{r_U=g\odot m_U}\qquad \text{先归一化完整 } v \text{ 再乘 mask，不再二次归一化}
$$

$$
t_{cond}=\mathrm{sg}(t),\qquad
\boxed{g^0=\mathrm{Norm}(E_I^0(I_a))}\qquad
\boxed{\hat g=D_\psi([r_U;t_{cond}])}
$$

$$
\boxed{L_{rec}=\frac1{|\mathcal V|}\sum_{i\in\mathcal V}\sum_{d=1}^{512}(\hat g_{i,d}-g^0_{i,d})^2}
\qquad
\boxed{L=10(L_{SIDM}+L_{DISM})+2L_{sparse}+\lambda_{rec}L_{rec}}
$$

**新增的两个组件：冻结初始化视觉参考（`reference_visual`）与训练期 decoder（`D_psi`）。**
`E_I^0` 是**加载共同初始化之后**从学生 visual 深拷贝出来的、永久冻结且处于 `eval()` 的独立模块；
`D_psi` 是本轮唯一新增的可训练模块。

| 配置 | 值 |
| --- | --- |
| decoder | `nn.Sequential(Linear(2*512→512), GELU, Linear(512→512))`，无 dropout / LN / 残差 / 输出归一化 |
| decoder 初始化 | 正常随机初始化，`decoder_seed = 0`，在可恢复 RNG 区间内 |
| `lambda_rec` | **1.0**，全程常数，无渐增、无动态平衡 |
| 三组 optimizer | 学生非 mask_net：AdamW lr 1e-6 wd 1e-2 warmup 200；mask_net：lr 1e-3 wd 0 warmup 0；**decoder：lr 1e-4 wd 0 warmup 0** |
| reference_visual | **无 optimizer**，fp32，`requires_grad_(False)`，`eval()` |
| LR horizon | 3 × len(loader) = 3651（三组同一 horizon） |
| 精度 | 学生 fp32 master + bf16 autocast；**参考塔、归一化、decoder、重建核心均为 fp32 且 autocast 关闭** |
| 数据 | Full ShareGPT4V，seed 0，256 pairs/GPU × 4 A800 = 1024 global pairs |
| mask | hard-ST，`rho = 0`，`duplicate_policy = exclude` |
| 预算 | `max_steps = 500` |

未包含（明确不做）：CVSSL 图像—图像对比损失、ExGAP、Global Absorption、Said+Unsaid 线性闭合、
Global 追逐 decoder 输出、额外 Global–Text CLIP loss、正交/能量/mask 稀疏度新约束、H11/H12 patch
pooling、batch 文本池语义合成、EMA/伪 caption/外部强教师。旧的 `gap_completion` 混合入口未复用。

## 2. 改动文件与实现 commit

| 文件 | 说明 |
| --- | --- |
| `model/said_cls_reconstruction.py` | 新增：`C1TrainModule`、`TextConditionedReconstructor`、`reconstruction_loss`、`visual_embed_dim`、`_isolated_rng`、`reference_fingerprint` |
| `train/train_said_cls_c1.py` | 新增：`said_cls_tcr` 训练入口、`c1_train_step`（三组 optimizer + 跨 rank 重建缩放）、输入依赖诊断、checkpoint |
| `tests/test_said_cls_c1.py` | 新增：13 个验收测试 |
| `tests/_c1_ddp_worker.py` | 新增：2-rank 真实 DDP 等价 worker |
| `tools/exp_said_cls_c1.sh` | 新增：单次 500 步训练 + canonical 评测 |
| `tools/diag/{c1_summary,export_c1_student,build_c1_table}.py` | 新增：日志摘要、裸学生导出与严格校验、对照表 |

**实现 commit**：见 `progress.json` 中的 `implementation_sha`（训练使用该 SHA 记录在每个 checkpoint 的
`git_head` 字段）。旧 C0 默认行为**未改变**：本 worktree 复用现有 `build_optimizers`、`cosine_lr`、
`Share4VCvsslDataset`、`compute_smartclip_terms`、`said_mask_from_hidden`，**未修改任何旧文件**。

## 3. 最小测试与真实运行状态

`pytest tests/test_said_cls_c1.py -q` → **13 passed**。覆盖：

| 测试 | 结论 |
| --- | --- |
| 尺寸/有限值，`r_U` 未被二次归一化 | PASS |
| `reconstruction_loss`：零预测 + 单位目标 = **恰好 1.0**（证明没有误除 512） | PASS |
| 无效样本跳过；全无效时返回**连通的零**且梯度有限 | PASS |
| 冻结参考 == 同一初始 visual；**无 storage alias**；`train()` 后仍 `eval()`；一步更新后权重未变 | PASS |
| rec-only backward：decoder 与学生 visual 非零；text/mask/reference 梯度为 0 | PASS |
| `lambda_rec = 0` 时 backward 标量 == SmartCLIP loss，decoder 梯度为 0，mask_net 仍训练 | PASS |
| 2-rank 真实 DDP：逐参数梯度 == 单进程参考（RTOL 1e-5），且 rank0/rank1 逐位相同 | PASS |
| 导出的裸学生 checkpoint 不含 teacher/decoder 键，`strict` 加载零缺失 | PASS |
| CLI/import 启动路径 | PASS |

**真实运行状态**：C1@500 **已完成**（`completed_steps = 500`，wall 708.7 s，1.341 s/step，峰值
28.36 GB，0 非有限值）。未执行部分在 §8 标注。

### 训练过程中暴露并修好的真实缺陷

1. `clip.embed_dim` 在 LongCLIP 的 `CLIP` 上不存在（只在构造参数里），桩模型掩盖了它 → 改为从
   `text_projection` 形状推导（`visual_embed_dim`）。
2. **`lambda_rec` 在 DDP 缩放里丢失**：`c1_train_step` 直接用 `loss_rec * scale` 回传，导致无论
   `lambda_rec` 设成什么都按 1 训练 decoder → 改为 `lambda_rec * loss_rec * scale`（测试已钉住）。
3. `rec_valid_count` 被转成 Python float 后参与 DDP 缩放 → 改为传递张量句柄 `rec_valid_count_tensor`。
4. `copy.deepcopy` 会消耗**可变数量**的全局 RNG → 参考塔与 decoder 的构造放进同一个可恢复 RNG 区间，
   避免污染数据/caption 流。
5. 启动脚本三处工程问题（argparse 不认识的 `--rho`、`conda run` 相对路径、worktree 推导），已修。

## 4. 冻结参考与精度检查（实测）

```
DTYPE_AUDIT rank=0 parameter_dtypes=[('decoder:torch.float32', 4), ('torch.float32', 317)]
  reference_dtypes=['torch.float32'] reference_requires_grad=[False]
  decoder_params=787456 amp_enabled=True amp_dtype=bf16 reference_eval=True
reference_visual_state_sha256 = 1428ea22801976c3358cf10905a64ba00bab7cadacda80e352016b3f8a2e1e9d
reference_visual_state_sha256_final = 1428ea22801976c3358cf10905a64ba00bab7cadacda80e352016b3f8a2e1e9d
reference_unchanged = True
```

* 参考塔 317 张量全部 **fp32**、`requires_grad = False`、`eval()`；
* 参考与学生 visual **无共享 storage**（启动时断言，共享即报错）；
* 500 步训练前后参考摘要**逐位相同** → 参考确实未被训练影响；
* 参考状态单独保存一次（`reference_visual_state.pt`，含 sha256 与来源），每个 checkpoint 记录路径与 hash；
* 学生 317 张量 fp32 + decoder 4 张量 fp32，全部在 DDP 内同步。

### 重建梯度路径（实测，非断言）

以 `lambda_rec = 1.0`、SmartCLIP 分支梯度用 hook 置零后测量：

| 目标 | 实测 | 判定 |
| --- | --- | --- |
| decoder | 非零 | 允许 |
| 学生 visual | 非零 | 允许 |
| text encoder / text_projection | **0**（可达但梯度为零） | 阻断 |
| mask_net | **0** | 阻断 |
| frozen reference | **0** | 阻断 |

**如实记录**：本版 `r_U = Norm(v) * m_U`，L2 归一化在 mask 之前，分母会耦合坐标，因此
**`dL_rec/dv` 在 Said 坐标不严格为零**，本报告不作此断言，也未为了让其为零而 detach 归一化分母。
共享 visual 参数导致的 Global 变化是**预期存在的风险**（见 §7 的
`native_global_to_reference_cos`：0.9996 → 0.9864）。

## 5. 重建诊断与常量预测基线（固定 64 图 cohort，no_grad，mask 只算一次）

| 干预 | step 1 误差 | step 100 | **step 500** | step 500 Δ |
| --- | ---: | ---: | ---: | ---: |
| `D(r_U, t)` 正常 | 1.3493 | 0.5783 | **0.4000** | — |
| `D(r_U, 0)` 去文本 | 1.3449 | 0.9212 | **0.7853** | **+0.3853** |
| `D(0, t)` 去 U | 1.3472 | 0.5928 | **0.4404** | **+0.0403** |
| `D(r_U, t_permuted)` 换错文本 | 1.3506 | 0.6357 | **0.5988** | **+0.1988** |
| `D(r_U_permuted, t)` 换错 U | 1.3480 | 0.5874 | **0.4473** | **+0.0473** |
| 常量预测基线（预测 teacher 均值 μ0） | 0.5671 | 0.5925 | 0.5683 | — |
| 特征 $R^2$ | −1.379 | +0.024 | **+0.296** | — |
| `pred` 跨图方差 | 2.19e-05 | 6.40e-05 | 2.76e-04 | — |

结论（仅限"decoder 是否对输入敏感"）：

* **decoder 确实学习了**：重构误差 1.349 → 0.400，$R^2$ 从 −1.379 升到 **+0.296**，且已明显优于
  常量基线 0.568（即不只是拟合一个共同方向）；
* **对文本高度敏感**：去掉文本使误差 +0.385（近乎翻倍），换错文本 +0.199；
* **对 `r_U` 敏感但弱得多**：去 U +0.040，换错 U +0.047 —— 两者量级相近，说明 decoder 用到了
  `r_U` 这个**输入槽位**，但"换成别的图的 U"造成的损失与"置零"差不多，**不足以证明它提取了
  该图特有的互补语义**；
* `pred` 跨图方差仍很小（2.8e-04，而目标是单位向量空间），提示预测仍偏向一个共享方向；
* 这些只是当前 decoder 的输入敏感性证据，**不等于**单模态重训后的最优性能，也**不证明语义解耦**；
  `r_U` 的 mask 模式本身携带 caption 条件痕迹。

## 6. 原生 CLS 检索（canonical，`phase30a-1d-fixed-cohort`）

主表示固定 `normalize(student_clip.encode_image(image))` / `normalize(student_clip.encode_text(text))`；
**不使用 decoder 预测向量、不使用参考教师检索、不做文本条件化 reranking**。导出裸学生 checkpoint 时
严格校验（`strict=True`，`missing=[] unexpected=[]`），评测时 `loaded 317 tensors, skipped 0`。

| 指标 | Initial | S0@500 | C0-L10@500 | C_L03@500 | **C1@500** | C1 − S0@500 |
| --- | --- | --- | --- | --- | --- | --- |
| COCO I2T R@1 | 0.5170 | 0.6058 | 0.6042 | 0.6046 | **0.6038** | **−0.0020** |
| COCO I2T R@5 | 0.7662 | 0.8220 | 0.8224 | 0.8254 | **0.8248** | +0.0028 |
| COCO I2T R@10 | 0.8428 | 0.8906 | 0.8896 | 0.8840 | **0.8880** | −0.0026 |
| COCO T2I R@1 | 0.3269 | 0.4124 | 0.4072 | 0.4128 | **0.4122** | **−0.0002** |
| COCO T2I R@5 | 0.5776 | 0.6709 | 0.6655 | 0.6715 | **0.6719** | +0.0010 |
| COCO T2I R@10 | 0.6823 | 0.7662 | 0.7636 | 0.7664 | **0.7664** | +0.0002 |
| 1K first I2T R@1 | 0.5440 | 0.6620 | 0.6670 | 0.6670 | **0.6620** | +0.0000 |
| 1K first T2I R@1 | 0.5140 | 0.6350 | 0.6220 | 0.6270 | **0.6340** | −0.0010 |
| 1K sparse I2T R@1 | 0.7460 | 0.9010 | 0.9030 | 0.9000 | **0.9050** | +0.0040 |
| 1K sparse T2I R@1 | 0.7400 | 0.8790 | 0.8840 | 0.8840 | **0.8830** | +0.0040 |
| 1K full I2T R@1 | 0.7580 | 0.9650 | 0.9640 | 0.9660 | **0.9690** | +0.0040 |
| 1K full T2I R@1 | 0.7760 | 0.9620 | 0.9620 | 0.9610 | **0.9630** | +0.0010 |

全部 24 格见 `outputs/cvssl_screening/c1_tcr/C1_step500_canonical.json`；差异范围
**−0.0026 ~ +0.0040**。

### 冻结晋级门（全精度）

```
COCO I2T R@1  0.6038000000 vs S0 0.6058000000   FAIL
COCO T2I R@1  0.4121600000 vs S0 0.4123600000   FAIL
J_C1 = 0.5079800000   J_S0 = 0.5090800000   delta = -0.0011000000
至少一项严格提升: False
判定: NOT_PROMISING
```

C1 在两个主方向上**都略低于** S0@500（−0.0020 / −0.0002），因此**不满足**
`PROMISING_FOR_LONGER_RUN`。差异幅度与"同配置重复运行的噪声"同量级，但按预设规则不能声称有改善。

## 7. 训练侧日志（每个训练步记录，节选）

| step | `loss_smart` | `loss_rec` | `cos(pred, ref)` | `pred_norm` | `ref_norm` | `native_global_to_reference_cos` | `r_U` 占比 | decoder_lr | sec/step | 峰值显存 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50.79 | 1.3758 | −0.0319 | 0.5819 | 1.0000 | 0.99959 | 0.5558 | 1e-4 | 9.12 | 27.20 GB |
| 20 | 9.06 | 1.0157 | 0.1678 | 0.3763 | 1.0000 | 0.99759 | 0.0794 | 1e-4 | 1.33 | 28.36 GB |
| 100 | 8.66 | 0.5552 | 0.6745 | 0.5683 | 1.0000 | 0.99034 | 0.1252 | 9.98e-5 | 1.32 | 28.36 GB |
| 500 | — | — | — | — | — | — | — | — | — | — |

（step 500 的完整记录在 `salu_log.jsonl`；`rec_valid_fraction` 全程 **1.000**，即没有空补集样本被跳过。）

`native_global_to_reference_cos` 从 0.9996 降到 0.9864：**学生原生 Global 已被辅助目标推动**，
这既解释了"Global 变化是预期风险"，也说明不能声称原生表示未受影响。

## 8. 未执行部分（明确标注）

| 项 | 状态 |
| --- | --- |
| C1 三 epoch | **NOT RUN**（按约定 STOP） |
| G0 / R0 | **NOT RUN** |
| 新的 C0 实验 / 超参搜索 | **NOT RUN** |
| 梯度夹角探针 | **NOT RUN**（未因缺该字段暂停） |
| `lambda_rec` / decoder 容量 / horizon 搜索 | **NOT RUN** |
| Unsaid 语义保留的独立评测 | **NOT RUN → NOT ESTABLISHED** |
| 完整历史 pytest 套件 | **NOT RUN**（只在 worktree 内运行了新增的 13 个 C1 测试；旧套件位于主工作目录，未因本轮改动而变更） |

## 9. 结论（四项分别回答）

1. **实现是否正确 —— 是。** 13/13 验收测试通过；冻结参考与共同初始化一致、无 storage alias、
   `train()` 后仍 `eval()`、500 步后摘要逐位未变；rec 梯度只到 decoder 与学生 visual，
   text/mask/reference 为 0；`L_rec` 归一化正确（零预测+单位目标 = 1.0）；2-rank 真实 DDP 梯度与单进程
   参考一致且跨 rank 逐位相同；导出裸学生可严格加载。训练途中发现并修好 4 个真实缺陷（含
   `lambda_rec` 在 DDP 缩放中丢失这一实质 bug）。
2. **重建是否学习 —— 是。** 重构误差 1.349 → 0.400，$R^2$ −1.379 → **+0.296**，优于常量基线
   0.568；去文本 +0.385、换错文本 +0.199，说明 decoder 真的在使用文本条件；去 U / 换错 U 仅
   +0.040 / +0.047，说明 `r_U` 的使用较弱且不特异。`pred` 跨图方差仍小。
3. **原生检索是否改善 —— 否。** 两个主方向都略低于 S0@500（−0.0020 / −0.0002），
   24 格差异在 ±0.004 内，判定 `NOT_PROMISING`。按预设规则**不能**声称检索改善。
4. **Unsaid 语义是否已证明 —— 没有，NOT ESTABLISHED。** 本轮没有 withheld/属性/概念评测；
   重建改善、$R^2$、输入干预敏感性都不是语义保留的证据。

**综合判定：`AUXILIARY_TASK_IMPROVED_ONLY`。** 重建目标确实被学习，但学生原生 CLS 检索没有改善，
因此本轮**不把 C1 称为成功方法**，也**不据此自动新增别的损失或调参**。本轮完成后 STOP：
不跑 C0 新实验、不调超参、不自动开始 C1 三 epoch。
