# PG-CLIP v0.1 — Pre-Projection Gated CLIP（原生对齐 + 投影前文本条件视觉选择）

- objective：`clip_native_preproj_mask`
- arm：`PG_CLIP_V01`
- phase：`pgclip-v0.1`
- branch / worktree：`codex/pgclip-v01` / `/root/SAID-pgclip-v01`
- base commit（本 worktree 建立时的实际提交）：`894599d`（C1/B500 线），实现提交见 `report.md`
- 共享初始化：`/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt`
  - 文件 SHA256：`c1a4a2be1b212f38677f729a4f03160d788a6d6db79ca5021a43022af0f5cba8`
  - 模型 state 摘要（张量值摘要，与文件 SHA 分开记录）：写入 run 的 `config.json` 与每个 checkpoint 的 `digests.initial_state_digest`
- 本轮只跑**一个**配置，不做任何参数搜索。

## 1. 精确取点

图像侧（一次视觉主干前向）：

```
h_i = ln_post( CLS_output_of_visual_block_12 )      # [B, 768]
v_raw = h_i @ clip.visual.proj                      # [B, 512]
```

- 第 12 个（最后一个）transformer block **之后**、原生 `ln_post` **之后**、`visual.proj` **之前**。
- 不是 `forward_prefinal` 的第 11 层隐藏表示，不是 patch token，不是先投影到 512 再扩回 768。
- 新接口 `CLIP.encode_image_with_preprojection(image)` / `VisionTransformer.forward_with_preprojection`
  与 `encode_image` 共用同一段 token 构造（`_token_sequence`），`ln_post` 与原生投影各只执行一次；
  测试用计数器验证 trunk 调用 1 次、`ln_post` 调用 1 次，并验证 `v_raw == h @ proj == encode_image(x)`。
- `encode_image` / `encode_text` 的默认行为与 `state_dict` 键集合完全不变（317 键基线文件逐行比对）。

文本侧（一次文本主干前向）：

```
H_j   = ln_final( full_token_sequence )             # [B, 248, 512]
eot_j = argmax(token ids)                           # 仓库既有 EOT 约定，不是非零 token 计数
t_raw = H_j[eot_j] @ clip.text_projection           # [B, 512]，FP32
```

- 不为拿 `H` 先算一次 bf16 投影再重算；`encode_text_final_hidden` 只做一次前向，投影由调用方在 FP32 下做一次。
- 输入截断检查：用参考 tokenizer 逐条计算 `len(encode(caption)) + 2 > 248` 的条数（`over_capacity_captions`），
  并记录有效长度（`argmax + 1`）的最大值。

## 2. 两条路径

```
W = clip.visual.proj                                 # 唯一一个 Parameter，两条路径共享，不复制、不独立初始化
原生路径      QG[i,j] = 100 * dot( Norm(h_i @ W),            t̂_j )
条件路径      QP[i,j] = 100 * dot( Norm((h_i * mask_j) @ W),  t̂_j )
t̂_j = Normalize(t_raw_j)，normalize eps = 1e-6，scale 固定 100
```

候选规则：固定图像 `i` 时每条候选文本 `j` 用**它自己的** `mask_j`；固定文本 `j` 时所有候选图像共用**同一个** `mask_j`；
正配与负配使用完全相同的计算规则；`mask` 只能来自候选 caption 本身，不能来自标签。

条件投影的范数必须精确：

```
||u_i,j||² = (h_i*m_j) @ (W @ Wᵀ) @ (h_i*m_j)ᵀ
```

`W @ Wᵀ` 非对角，因此**不允许**用 `sum(h² * m²)`（投影前对角范数，TriMask 的 512 维公式）替代；
实现采用分块显式 `masked_h @ W` + `F.normalize(..., eps=1e-6)`，块大小默认 `image_chunk=32`、`text_chunk=64`，
`[B_global, B_global, 768]`（或 `…,512`）激活永不常驻；块内可选非重入梯度检查点
（`use_reentrant=False, preserve_rng_state=True`），块内没有任何 collective，输出块保持可导。

## 3. 独立 768 维文本条件门

```
stem  = MaskNetwork(width=512, layers=1, heads=8)     # 仓库既有实现：1 层 ResidualAttentionBlock
pool  = AttentionPool(512)                            # 既有 softmax-over-tokens 汇聚，整条 248 token 序列参与
a_j   = Linear(512, 768, bias=True)(pool(stem(H_j.detach())))
p_j   = sigmoid(a_j)
hard_j = (p_j >= 0.5)  （前向 0/1）
mask_j = hard_j + (p_j - p_j.detach())                # 前向硬 0/1，反向走 sigmoid 的直通梯度
```

- 初始化：stem 正常初始化；`projection.weight = 0`；`projection.bias = log(8)`；因此初始 `p = 8/9`、`mask ≡ 1`
  ——所有候选文本从同一个“全开”选择出发。只把输出层置零，绝不把 stem 也置零。
- 初始一步 stem 梯度为 0（因为输出层权重为 0），输出层权重更新后 stem 即获得梯度；测试固定该行为，
  不把它误判为死分支。
- 模块构造在隔离、可恢复的随机状态中完成（Python / CPU / 当前 CUDA 设备 RNG 全部恢复），
  不污染主模型初始化与数据随机流。
- gate 注册在 TrainModule 外层；`clip` 本体只注册一次；`W` 只通过 `clip.visual.proj` 引用。
- 旧 `clip.mask_net`：为裸学生 state 兼容保留，但**冻结、不前向、不进优化器、不产生旧稀疏项**。
- `clip.logit_scale`：不参与评分（评分用固定 100 倍），保留兼容键并冻结/排除优化器。

## 4. 损失

```
LG = CE(QG, y) + CE(QGᵀ, y)          # 双向相加，不取平均
LP = CE(QP, y) + CE(QPᵀ, y)
LS = mean(|mask|)                    # 本地 caption × 768，每步只算一次，用带 ST 计算图的 mask
L_total = 5*LG + 5*LP + 1*LS         # 固定 5/5/1，不乘 0.5，不在总 loss 外再乘任何系数
```

初始 `mask ≡ 1` 时 `QP = QG`、`LG = LP`、`LS = 1`，因此 `L_total = 10*LG + 1`；
共享编码器/投影的梯度对应 `10*LG`（gate 仍有条件对齐与稀疏梯度）。**不得**把 `L_total` 的数值说成等于 `10*LG`。

## 5. 梯度职责

| 项 | 更新视觉/文本主干与两侧原生投影 | 更新 gate |
|---|---|---|
| `LG` | 是 | **否**（测试：gate 梯度全为 None） |
| `LP` | 是 | 是 |
| `LS` | **否**（`H` 已 detach） | 是（测试：主干梯度全为 None） |

视觉 `h` 与 `W` 在条件路径中都 live，候选文本 `t` 也 live；不是只训练门头的冻结主干探针。
本版只允许文本控制 768 个视觉隐藏成分的**乘法保留/排除**：不做 `(h*m)@W + A*t`、不做文本条件加性 bias、
不生成 768×512 动态矩阵、不做交叉注意力。保留原生无条件前向接口**不等于**保证原生能力不变。

## 6. DDP 路线

每步：本地图像/文本各编码一次 → 本地 caption 生成一次 mask → 可导 gather 全局文本向量与 mask →
本地图像行 × 全局文本得到 `QG_local/QP_local` → 可导按行 gather 得到全局标量矩阵 →
I2T 用本地图像行、T2I 用属于本 rank 文本的列 → `LS` 用本地 mask 均值。

- 采用：本地 anchor mean + autograd-aware gather + 标准 DDP 参数梯度平均；**不额外乘/除 `world_size`**。
- 由独立单进程参考与真实两 rank 一步更新测试确认（`tests/test_pgclip.py::test_f_*`）。
- 远端文本、远端 mask、全局评分都不 detach；`W` 在所有相关路径中接收梯度。
- 训练只走 `ddp_model(...) → backward() → optimizer.step()`；`.module` 仅用于参数分组、读状态、导出与日志。
- 大 feature gather 之前一致检查各 rank batch 大小（不等长共同报错），本轮不新增 ragged gather。
- 每个优化器步之前检查梯度有限性（各 rank 共同确认，`all_reduce(MAX)`）；任一卡失败则拒绝更新并退出，
  不加 clipping、不 `nan_to_num`、不跳过异常更新。

## 7. 数据、优化器与预算

- 数据：现有 Full ShareGPT4V 清单（`[1000:]` 切片）、`image_a` 标准视图、原有前缀采样
  （`caption.replace('\n',' ')` → `split('. ')` → `k = random.randint(1, len(sentences))` → 前 k 句）、`seed=0`、248 token。
- 复用 `Share4VCvsslDataset(augment_view_b=False)`：视图 b 的随机生成器是 stateless 且被跳过，因此不消费全局随机流；
  实际 caption/sample/image_id/prefix_k 流摘要写入 `run_summary.json` 与 checkpoint。
- 只编码一个图像视图，不新增增强任务。
- 4 卡 × 256 pairs，global batch 1024，grad accumulation 1，恰好 500 次 optimizer update。
- 两个 AdamW：CLIP 组 `lr=1e-6, wd=1e-2, warmup=200`；gate 组 `lr=1e-3, wd=0, warmup=0`；`betas=(0.9,0.999)`，`eps=1e-8`。
  按 parameter id 验证：组间不重叠、覆盖全部可训练参数、旧 mask/logit_scale 已排除、共享 `W` 只出现一次。
- LR horizon：`3 * len(loader)`（按真实 loader 动态计算，预计约 3651，不硬编码）；`--max_steps 500` 只截断运行。
- checkpoint：0、20、100、250、500（step 0 在第一次更新前写出）；中间 checkpoint 不自动评估。
- 精度：视觉/文本主干 bf16 autocast；`h`、`H` 进入核心后转 FP32；原生投影、条件投影、最终文本投影、gate、
  normalize、score、loss 核心全部显式 FP32；所有参数保持 FP32 master。

## 8. 验收测试（A–G）

| 组 | 内容 | 位置 |
|---|---|---|
| A | 取点正确、单次前向、`v_raw == h@W == encode_image`、`H` 单次、EOT=argmax、state 键不变 | `tests/test_pgclip.py` |
| B | 全开初始化（`p=8/9`、`mask≡1`、`QP=QG`、`LG=LP`）、共享参数梯度 = `10*LG`、总 loss 值 = `10*LG+1` | 同上 |
| C | 分块 `QP` 与逐 pair 循环一致（含 h/W/t/gate 梯度）、非对角 `W@Wᵀ` 手工反例、禁用对角范数 | 同上 |
| D | 前向 0/1 且保留数不固定、候选列对应、`LG/LP/LS` 梯度职责、零初始化输出层与 stem 梯度行为 | 同上 |
| E | 全开/部分/全关/近零投影向量边界、数值有限、不偷偷补开或丢候选、投影能量比可 > 1 | 同上 |
| F | 真实两 rank DDP 一步与单进程同全局 batch 的梯度/更新一致（非全开 mask） | `_pgclip_ddp_worker.py` |
| G | 真 CLI 参数解析、小型真实 forward/backward、严格存取与裸学生导出（无 gate、key 集合一致、导出前后输出逐位一致） | 同上 |

## 9. 明确没有的东西

原 SmartCLIP 投影后 512 维 mask 对齐；TriMask 三路目标；文本 mask；U / reference / teacher / decoder / 重建；
交叉注意力；额外全局蒸馏；优势 loss、正交或能量正则；任何超参搜索。辅助（条件）分支只在训练时使用，
它**仍然参与训练共享视觉主干与原生投影**，因此不能声称它对原生表示没有影响。

## 10. 评估（500 步后，仅两项）

- 导出裸学生：`clip.state_dict()`；`PreProjectionGate` 不进入裸学生 state；旧 `clip.mask_net` 兼容键保留但不用于主检索。
- 严格加载：`missing/unexpected` 为空，键集合与原生 CLIP 预期一致；导出前后同精度、同原生接口输出一致。
- 主评估只用 `Norm(student.encode_image(I))` 与 `Norm(student.encode_text(C))`；不用 gate、不做两路融合、不做 reranking。
- COCO canonical（5000 图 × 25000 caption）与 Urban-1k（1000 × 1000），I2T/T2I R@1/5/10，沿用既有精度与预处理协议。
- 晋级门（照冻结定义）：COCO I2T R@1 ≥ 0.6058 且 T2I R@1 ≥ 0.41236，且至少一项严格提高 → `PROMISING_AT_500`，否则 `FAIL`。
- 本轮**不**运行 native-only 或 native+post-projection 控制臂，因此即使提升也只能归因于 PG-CLIP 这一整套组合，
  不能单独声称“投影前位置优于投影后”已被证明。
