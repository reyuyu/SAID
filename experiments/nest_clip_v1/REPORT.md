NEST-CLIP v1 第一轮实现与四卡冒烟验收（2026-09-29）

A2、A3 已顺序完成各 5 次真实同步更新，最终 torchrun 均返回 0。没有启动正式 500 步训练或正式检索评测。不能据此判断方法效果。

代码基于 `reyuyu/SAID` 的 `b6ced1ff80635a7671970b702f989da84cc721d5`。开始时工作区干净；本轮仅新增文件，未改动原 S0/Clean/Full 入口，本轮代码及小体积证据统一发布在 `codex/nest-clip-v1` 分支。实际新增源码 SHA256 和两组训练时的源码 SHA256 见 [最终审计](evidence/final-audit.json) 与各组 `config.json`。

新增文件：

- `model/nested_semantic_mask.py`：共享 backbone/mask 的完整 DDP forward；三个独立文本编码；FP32 评分与损失；可导全局候选；有限零梯度通信路径；软包含。
- `train/nested_semantic_data.py`：原 tokenizer 的未截断长度检查、固定句段代理、248-token 预打包、完整数据的 spawn 安全索引。
- `train/train_nested_semantic_mask.py`：四卡断言、NCCL 测试、完整日程、两组 AdamW 参数、检查点/RNG/日志和逐参数跨 rank 核对。
- `configs/nest_clip_a2.json`、`configs/nest_clip_a3.json`：仅 `arm` 不同；方法上仅 A3 加 `min(1,s/200)*L_inc`。
- `tests/test_nested_semantic_mask.py`、`tests/nest_ddp_worker.py`：单元、原实现回归和两卡参考测试。
- `tools/nest_clip.py`、`tools/eval_nest_native.py`：原始初始化、严格导出/加载核对、复用原生检索协议。
- `experiments/nest_clip_v1/run.sh`：显式选择正式训练、导出或单项评测；无默认训练动作。

单机四张 **A100 80GB PCIe**，nvidia-smi 总显存各 81920 MiB，精确查询的空闲显存各 81038 MiB（含驱动保留空间口径）。MIG Disabled，全部卡间 PIX，没有 NVLink。初检无计算进程，CUDA_VISIBLE_DEVICES/NCCL 环境变量均未设置；本轮未覆盖 GPU 可见性、未修改 MIG、未关闭 P2P/IB、未强制 loopback。PyTorch `2.5.1+cu124`、CUDA runtime `12.4`、NCCL `2.21.5`、驱动 `550.127.05`。环境原始记录见 [environment.json](evidence/environment.json)。

四 rank 的设备与最终成功运行 PID 如下；两组每 rank batch=256、WORLD_SIZE=4、accumulation=1，每个 rank 均完成 5 次更新。

| Rank / LOCAL_RANK / GPU编号 | A2 PID | A3 PID | GPU UUID |
|---|---:|---:|---|
| 0 / 0 / 0 | 774760 | 777428 | GPU-6810482c-0a4f-2547-407d-7b11db16655d |
| 1 / 1 / 1 | 774761 | 777429 | GPU-a85d54c4-620e-6813-9176-ed449a61c7a4 |
| 2 / 2 / 2 | 774762 | 777430 | GPU-251127fd-2cb2-c90d-6151-54360a5fcb82 |
| 3 / 3 / 3 | 774763 | 777431 | GPU-cea2e93a-3a77-7373-2c22-cfc59c5094e8 |

共同初始化为 服务器保存的 `shared/step000000.pt`（见[检查点清单](manifests/checkpoints.json)），SHA256：

```text
54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6
```

由 seed=0 下原 `longclip.load_from_clip('ViT-B/16')` 构造，原始 CLIP 文件 SHA256 为 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`。保留原 MaskNetwork 初始化以及冻结位置参数/可训练残差位置参数。没有使用历史训练权重。两组保存的 step-0 模型和 optimizer 与共同初始化逐项相同；CPU RNG 与共同初始化相同，各 rank 的 Python/NumPy/CPU/CUDA RNG 在 A2/A3 间一致。两组 5 步全部 sample/text/token 流摘要一致。

原数据 `share-captioner_coco_lcs_sam_1246k_1107.json` 有 1,246,901 条，按原顺序跳过前 1,000 条后为 **1,245,901** 条。没有因局部视图无效删图，也没有缩成 pilot 子集。每 rank sampler 长度 311,476，全局 padding=3；每 epoch **1,217** 次更新，尾批每 rank 180；完整 **H=3,651**。停止点和 horizon 分别写入检查点。真实冒烟只将 max_updates 设为 5；正式命令设为 500。

全文换行替换为空格，按字面 `. ` 分段，去空片段；穷举连续开头组合的实际 tokenizer 长度，选择 F/O/E 均不超过 248（含 SOT/EOT）的最长组合。E 保留可见 F 的末句。首句超长才显式截断原 caption 并禁用 O/E；单可见句也只参与 F。两组本次各观测 5,120 条训练样本，局部有效比例均为 **100%**，无回退、无重复图像 ID；此比例仅描述冒烟样本，不冒充全数据统计。索引元信息和原注释文件哈希见 [训练数据元信息](manifests/training_data.json)。

| 更新 | A2 全局 loss | A3 全局 loss | F候选 | O/E候选（各） | A3包含权重 |
|---:|---:|---:|---:|---:|---:|
| 1 | 56.216709 | 56.216709 | 1024 | 1024 | 0.000 |
| 2 | 24.234295 | 24.234529 | 1024 | 1024 | 0.005 |
| 3 | 22.777321 | 22.777306 | 1024 | 1024 | 0.010 |
| 4 | 21.252405 | 21.239073 | 1024 | 1024 | 0.015 |
| 5 | 18.103582 | 18.107388 | 1024 | 1024 | 0.020 |

日志 loss 是各 rank detached sums 归约后的全局值。完整双向 loss、稀疏/keep ratio、全开/全关率、软/硬包含、O/E IoU、梯度范数/有限性、学习率与逐步耗时见 [A2 日志](smoke/A2/steps.jsonl) 和 [A3 日志](smoke/A3/steps.jsonl)。全部 loss/梯度有限，未发现全开或全关样本。

| Rank | A2峰值 allocated/reserved GiB | A3峰值 allocated/reserved GiB | A2耗时秒 | A3耗时秒 | 两组末尾相对rank0最大参数差 |
|---:|---:|---:|---:|---:|---:|
| 0 | 17.301 / 18.475 | 17.301 / 18.475 | 24.829 | 24.774 | 0 |
| 1 | 17.301 / 18.475 | 17.301 / 18.475 | 24.830 | 24.774 | 0 |
| 2 | 17.301 / 18.475 | 17.301 / 18.475 | 24.830 | 24.774 | 0 |
| 3 | 17.301 / 18.477 | 17.301 / 18.475 | 24.830 | 24.774 | 0 |

耗时为训练循环、worker 启动、末点保存及一致性检查的合计，不含初始模型构造。逐步最大更新耗时之和 A2=12.450 秒、A3=12.784 秒。峰值来自每个进程的 CUDA allocator 统计。两组均使用相同的非重入编码器 block checkpoint 和 FP32 分块评分，未改变 batch、候选池或累积次数。

验收测试：

- **6 项单元测试通过**：文本边界/末句/token 预算、collate、包含梯度及权重、全开/全关数值、无重复参数、原 SmartCLIP 回归。原实现同输入单视图、稀疏系数2：loss/align/mask 最大差为0，梯度最大绝对差 `2.4414e-4`，SGD 一步参数最大差 `1.1921e-7`。梯度容差 `atol=3e-4, rtol=3e-5`。见 [测试日志](evidence/unit-final-details.log.txt)。
- **两卡 NCCL、8 个参考场景通过**：A2/A3 分别覆盖局部有效数不同、rank0 零有效、全局 V=0、V=1。对照独立单进程全局 batch（显式 pairwise 评分），包含可导 gather backward 与 DDP 平均。loss 最大差 `6.1035e-5`，梯度 `2.1362e-4`，SGD 一步参数 `5.9605e-8`。见 [逐场景结果](evidence/ddp-results.json)。
- **AdamW 数值限制明确记录**：原回归/两卡测试中，注意力 key bias 和池化 softmax 标量 bias 是理论零梯度方向；FP32 抵消误差可被 AdamW 的 `eps=1e-8` 放大。`lr=1e-4` 时最大参数差分别 `1.9712e-4` / `1.9896e-4`。仅这些明确的零方向允许 `atol=2.01e-4`，并要求其梯度绝对值 `<1e-4`；其余 AdamW 参数 `atol=5e-6, rtol=3e-5`。未裁剪或修改训练梯度，不宣称原实现任意输入逐位等价。
- **真实四卡通信通过**：每组启动时四 rank 的 all_reduce(1,2,3,4)=10；每 rank 对可导全局 gather 施加权重 rank+1，返回本地梯度均为10。两组完整训练采用一个 DDP、NCCL、4个不同UUID，四 rank 均前后向并更新5次；末尾逐参数 broadcast 比较差异全部为0。
- **两组导出/恢复核对通过**：step5 checkpoint 和裸学生严格加载；optimizer state 的 step 均为5；两张真实数据图像及文本的 native encode_image/encode_text，与训练模块 native 接口的最大差均为0。见 [A2导出核对](evidence/A2-export-check.json)、[A3导出核对](evidence/A3-export-check.json)。未将冒烟权重用于正式起点。

正式命令如下，均已准备但**未执行**。`run.sh train500` 展开为 `torchrun --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 -m train.train_nested_semantic_mask`，完整参数见 [run.sh](run.sh)。输出目录必须全新，避免覆盖结果；A2/A3 必须顺序运行。

```bash
cd /root/lk_projects/SAID
bash experiments/nest_clip_v1/run.sh train500 A2
bash experiments/nest_clip_v1/run.sh train500 A3

bash experiments/nest_clip_v1/run.sh export500 A2
bash experiments/nest_clip_v1/run.sh export500 A3
bash experiments/nest_clip_v1/run.sh coco A2
bash experiments/nest_clip_v1/run.sh urban A2
bash experiments/nest_clip_v1/run.sh coco A3
bash experiments/nest_clip_v1/run.sh urban A3
```

正式训练输出至 `/root/lk_projects/SAID-nest-clip-v1/formal/{A2,A3}`，保存 step0、每100步和step500。评测严格加载 step500 裸学生，使用 `normalize(encode_image) @ normalize(encode_text).T`；COCO 沿用 canonical similarity_chunk=512 和每图前5条描述，Urban-1k沿用原协议，均输出双向 R@1/5/10，不使用 mask/融合/rerank。未来按 A3−A2 的百分点差比较。

已解决的调试问题及范围：首次 A2 在第1步后因日志元数据默认 collate 报错；已接入专用 collate 并从共同初始化重跑。下一次 A2 完成5步且返回0，但退出阶段有 NCCL 异步清理诊断；补充退出前显式 CUDA 同步后，最终 A2/A3 均正常退出，无该错误。原失败/诊断日志和目录保留在 `evidence/`、`smoke/A2-failed-log-attempt`、`smoke/A2-teardown-diagnostic`，不作为最终验收结果。没有 OOM、未缩 batch、未终止其他人的进程。沿用的训练资产清单注明 SAM 图像未与 Meta 原始图像逐字节核对，此来源限制未被本轮消除。正式500步效果与检索结果按本阶段要求留待后续；目前无阻塞项。

GitHub 发布内容包括代码、配置、逐步训练指标、四卡验收、测试与调试日志、环境和初始化元信息。数据集、完整索引和大体积 checkpoint 保留在服务器；检查点路径、大小和 SHA256 见 [checkpoints.json](manifests/checkpoints.json)。上传的原始证据逐文件保持字节一致，校验清单见 [published_evidence.json](manifests/published_evidence.json)。
