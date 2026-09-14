# 方法、参数与数据流

## 模型和前向

- CLIP ViT-B/16：输入 224×224，视觉 width 768 / 12 层 / patch 16；文本 width 512 / 12 层 / 8 heads；共享特征维 512。tokenizer 为本仓库 LongCLIP，context length=248，`truncate=True`，EOT=49407。
- 原 S0 mask 是 `model/model_longclip.py::MaskNetwork`：一个文本 residual attention block 与 attention pooling。输入是当前前缀的完整文本 hidden state 的 detached 版本。保留原 hard straight-through sigmoid mask。
- 新 gate F：`Linear(1024,512) → GELU → Linear(512,512)`，共 787,456 个参数。初始化 seed=0，首层 Xavier uniform / bias=0，末层 weight=0 / bias=log(8)，因此初始概率=8/9，初始 hard mask 全开。构造 F 会恢复它触及的全局 RNG。
- 每样本只编码一次 `image_a` 和前缀；存在至少两个全局有效后缀时另行编码后缀。`image_b` 虽由原 dataset 生成，但不进入本训练目标。
- CLIP 原 positional embedding 经过本仓库的 77→248 插值后冻结；`positional_embedding_res` 可训练。必须使用固定代码和共同初始化，不替换成其他 long-context 实现。

## 候选条件和梯度

令 `g_i=normalize(encode_image(I_i))`、`tR_j=normalize(encode_text(R_j))`，`mS_j` 来自候选 j 的前缀。

```
x_ij = concat(stop_grad(g_i), stop_grad(g_i) * stop_grad(mS_j))
p_ij = sigmoid(F(x_ij))
mU_ij = float(p_ij >= 0.5) + p_ij - stop_grad(p_ij)
u_ij = normalize(g_i * mU_ij)
Q_ij = 100 * dot(u_ij, tR_j)
```

条件始终是候选列的 `mS_j`。F 不读取后缀 tokens、后缀 embedding、标签或文本特征拼接。F 的条件输入不把梯度传回 g/mS，但 `u_ij` 的视觉乘法路径仍向共享视觉编码器传梯度；后缀文本编码器通过可导 gather 收到梯度。

后缀归一化、F、score 和 CE 使用 FP32、eps=1e-6，外层编码器使用 BF16 autocast。评分固定乘 100，不用可学习 temperature。

## 损失和 DDP

```
L_S0 = 10 * (CE_SIDM + CE_DISM) + 2 * mean(abs(mS))
L_total = L_S0 + 1.0 * L_suffix
```

新 U 的稀疏系数=0；原 S0 稀疏项系数=2 保持不变。没有新增 loss 或 suffix lambda warmup。

全局候选按 rank 顺序拼接，标签 `rank * local_batch + arange(local_batch)` 不压缩。无效后缀对应候选列填 -inf，查询只选有效行；两个方向都按固定标签求 CE sum。每 rank 的 suffix backward loss 为 `W/V * (i2t_sum + t2i_sum)`，DDP 对参数梯度的平均得到全局平均目标。日志 `loss_suffix_global` 是全部 rank 的双向 CE sum 除以 V。日志中的 `loss_total`、`loss_s0`、`loss_suffix` 是 rank0 的训练标量，不应冒充额外归约后的全局总 loss。

V=0/1 时后缀损失为连接计算图的有限零，跳过 suffix encoder/gate；旧 S0 正常训练。`image_id` 用于数据追踪，本实现不额外按重复图片 ID 删除候选。尾批各 rank 大小相等，仍按该批实际 local_batch 计算标签。

后缀 tR 和 score 行的可导 all_gather 使用一维 contiguous 通信再恢复行形状，保留 autograd。该修复不修改旧 S0 helper，不 detach 特征、不加自定义 backward、不手动额外平均梯度。非全开 gate 的 Gloo/NCCL 数学证据位于 `evidence/ddp_total_*.json`。

## 数据和 K

原文件 `share-captioner_coco_lcs_sam_1246k_1107.json` 共 1,246,901 行，保持 JSON 原顺序，跳过最前 1000 行后训练 1,245,901 行。`--total-len 1000` 的含义是这个切片起点，**不是仅训练 1000 条样本**。

来源数量：COCO train2017 118,287；LAION/CC/SBU 的 LLaVA pretrain 图 558,128；SAM 图 569,486。完整来源统计与句子片段数量直方图见 `manifests/training_data.json`。

每次取样：将 caption 中 `\n` 替换为空格，以字面字符串 `. ` 分割得到 parts；用原 dataset 的 Python `random.randint(1,len(parts))`（两端包含）抽 K。前缀是 `'. '.join(parts[:K])`。suffix 使用 K 之后、最后一个非空句子之前的非空片段拼接；末句排除，末尾空片段不构成句子。后缀 tokenize 后 EOT 位置 >1 才有效。不为增加 V 重抽 K，不把分句替换成其他自然语言分句器。

- `DistributedSampler(shuffle=True, seed=0, drop_last=False)`，每 epoch 调 `sampler.set_epoch(epoch)`。
- `DataLoader(num_workers=8, persistent_workers=False, pin_memory=True, drop_last=False)`，每 rank 8 workers，共 32。默认 prefetch_factor=2，默认 worker seed 规则，未提供自定义 generator/worker_init_fn。更改 worker 数会改变 Python K 的随机流。
- 每 rank sampler 长度 311,476；1217 batches/epoch。前 1216 批每 rank 256，最后一批 180，所以最后一批 global batch=720。sampler 每 epoch 补齐 3 条，保持原行为；绝大部分批次 global batch=1024。
- 每 epoch 总采样次数 1,245,904，3 epoch 合计 3,737,712；这是含 sampler padding 的样本消费次数。3 epoch 恰好 3651 次更新。
- 本 trainer 没有调用 `dataset.set_epoch`；dataset 的 view-b epoch 字段保持 0。这是现有实现的实际行为，复现时不要擅自改变。view-b 不参与 loss。

`image_a`：PIL RGB，Resize(224,BICUBIC) 保持长宽比，CenterCrop(224)，ToTensor，CLIP mean `(0.48145466,0.4578275,0.40821073)` / std `(0.26862954,0.26130258,0.27577711)`。view-b 用同一窗口，独立 stateless generator 的轻微缩放与 GaussianBlur；不会消耗前缀的 Python RNG，但仍有 CPU 加载成本。

## 全部训练配置

| 参数 | 值 |
|---|---|
| suffix_mode / run_type | masked / formal |
| world_size / 单卡 batch | 4 / 256 |
| accumulation / seed | 1 / 0 |
| epochs / horizon | 3 / 3651 |
|阶段总步数上限|500，1000，3651；都是累计更新数|
| base_model | B16（ViT-B/16） |
| CLIP LR / weight_decay | 1e-6 / 1e-2 |
| 原 S0 mask LR / weight_decay | 1e-3 / 0 |
| 新 suffix gate LR / weight_decay | 1e-4 / 0 |
| AdamW（全部三组） | betas=(0.9,0.999)，eps=1e-8，amsgrad=False，maximize=False，默认 foreach/fused 选择（未显式设置） |
| warmup | CLIP 200 updates；mask 与 suffix 0 |
| precision | FP32 master parameters + BF16 autocast；无 GradScaler |
| chunks | image_chunk=16，text_chunk=32 |
| clipping / activation checkpoint | 未使用梯度裁剪；未使用 activation checkpoint |
| DDP | NCCL，find_unused_parameters=True，static_graph=False，默认 bucket 设置 |
| save_every | 100；另保存初始化和阶段最终 checkpoint |

LR 在每次更新之前以已完成步数 `s` 调用。CLIP `s<200` 时为 `base_lr*(s+1)/200`；之后为 `base_lr/2*(1+cos(pi*(s-200)/(3651-200)))`。另外两组为 `base_lr/2*(1+cos(pi*s/3651))`。step1000 对应 s=999。续训不重置 warmup，不将 horizon 缩成 500 或1000，不因暂停测评重新起算。

原机器 4×A800-SXM4-80GB，driver 535.129.03；Python 3.10.21、PyTorch 2.5.1+cu124、torchvision 0.20.1+cu124、cuDNN 9.1。完整版本见 environment。TF32 matmul=False，cuDNN allow_tf32=True，cudnn benchmark=False，未启用 deterministic algorithms。NCCL loopback 设置为 `NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 GLOO_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=0,1,2,3`；torchrun 为多进程默认设 OMP_NUM_THREADS=1。

## 保存和恢复

完整 checkpoint 包含 clip_state、suffix_gate_state、三组 optimizer_states、completed_steps、epoch、step_in_epoch、config、源码 SHA 与状态 digest。每100步保存；最终3651也保存。bare student 仅含 clip_state，不能用于续训。

恢复需要原完整 checkpoint、完整连续的 `salu_log.jsonl`、同初始化路径和固定配置。新代码从 completed_steps 推导下一批；旧500最终 checkpoint 的游标曾多前移一批，已按实际更新计数恢复。回放所有已消费 batch 以恢复 worker/K 随机流，四 rank 的 sample/prefix/suffix 累计 SHA256 必须与 checkpoint 匹配，再执行下一次更新。回放不会做 forward/backward/optimizer.step；耗时是正常的启动成本。

checkpoint 未保存任意外部 RNG 的通用快照；此恢复机制针对本固定、无 dropout 随机前向的模型和原 loader。不要将其当作支持任意随机增强/架构变化的通用 resume。实际 main loop 的小模型跨 epoch 对照、旧一批偏移反例和 optimizer/LR/stream 一致性证据位于 `evidence/resume_main_evidence.json`。

## 原生评估

固定 frozen evaluator 副本及 SHA 见 `reference_eval/`。COCO：torchvision CocoCaptions 的排序与标注顺序，每图前5条 caption，5000图/25000文本；图像每批64，对应文本每批最多320；特征转 CPU FP32，similarity_chunk=512，逐行 argsort 的 R@1/5/10 语义固定。不要改成批量 topk 或全矩阵比较来声称完全相同。

Urban：按图片 stem 排序，与同 stem `.txt` 首行配对，1000图/1000文本；图像每批64，**1000条文本一次编码**，GPU原生归一化特征及 full 1000×1000 pool，`topk`，对角为正样本。COCO 与 Urban 的排序/文本分批不同，不能统一替换成另一个评估器。
