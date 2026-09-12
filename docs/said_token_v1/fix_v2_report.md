# T1 fix_v2: implementation and validation

The audited implementation detached its positive ranking baseline and used an incorrect router proxy.
Historical `t1_500step` artifacts remain untouched and are INVALID_FOR_T1_METHOD_COMPARISON.
Starting HEAD was 39612af4a7047933beae9153b4edaa2acdf1704b (clean); its only change since audited
8bdb632f1465bd37bce3f332e731febea807688a was the stopped-run ledger. No old T1/GPU job needed stopping.

## Corrections and mathematical mapping

- `ranking_sums` subtracts live matched scores in both hinges. Autograd-aware gather preserves global
  positives. Active 3x3 counterexample: correct SUM loss=.3, diagonal score gradients=-2/3,
  off-diagonal=1/3, common-shift derivative approximately zero. U gate remains detached.
- `soft_from_grid` uses C.detach(), logsigmoid(a), CLS log-weight=0, live logsumexp numerator and
  logsumexp(log_w) denominator. Constant C=c gives 2c and approximately zero gate gradient.
  Nonconstant independent weighted-exponential reference and FP64 finite differences pass.
- Separately restore the approved directional sum: remove the historical 0.5 in Said. Pair score
  also remains the sum of both token matching directions. L_total=L_Said+0.1*L_rec.
- FP32 numerical cores explicitly disable autocast; encoder outputs remain bf16, master weights FP32.
  U validity uses raw norm>1e-6 and nonzero actual complement count. Nonfinite inputs/features/target/
  prediction/loss fail rather than being repaired. Empty captions match EOS only.
- `PairwiseScorer.from_prepared` shares one cosine grid for hard and proxy directions. Matched positive
  path uses [b,33,33]. Local image rows x all global texts, live gathered candidate/query/positive
  gradients, and detached metadata retain all 1024 candidates. A symmetric legal set excludes identical
  image IDs and identical effective caption sequences, ignoring padding. M counts ordered pairs once.
- Backward Said=W*(local_I2T_sum+local_T2I_sum)/M; Rec=W*local_rec_sum/V, weighted 0.1 once.
  Detached sums produce true global logged means. First full batch M=1,047,550 (two extra duplicate
  exclusions beyond the 1024 diagonal pairs); local rank0 pairs=261,887.
- Token normalization/router projection happen once. Non-reentrant block checkpointing contains no
  collectives. 32x64 chunks give 128 blocks/rank at 256x1024; no repeated full global image grid.
  Intra-image slot cosine and all-image slot cosine use vector-sum identities, not 8192x8192 matrices.

## Validation

41 targeted/existing tests pass (37 unchanged tests in combined run, four DDP tests rerun after final
health-probe fix). Includes independent full small reference, scores/parameter gradients, matched
diagonal, one-grid count, checkpoint/chunks, exact duplicates, padding, empty text, invalid U, and
real 2-rank backward synchronization. Unequal local sizes fail before large gathers. No-loss rows keep
connected zero. Rec leaves text/router/frozen teacher unchanged; Said trains both backbones/modules.

FP64 continuous gradcheck: eps=1e-6, atol=1e-7, rtol=1e-5. FP32 independent parameter max error bound
2e-6+3e-5*max|reference|; DDP 2e-7+2e-5*max|reference|. No loss-factor or zero-positive failure is hidden.
The normal AdamW test verifies gradients and identical synchronized rank updates. Degenerate fixture
SGD updates verify single/DDP parameter equivalence because Adam's eps amplifies rounding in exactly
zero softmax-bias directions; production remains the original AdamW. No algorithm hyperparameters changed.

Actual CLI: two real-data GPU steps saved 0/1/2 with logs and unchanged teacher. This plumbing check used
small batch and is not a method result. Short full-configuration profile used four A800, 256/rank,
global1024, 32x64 chunks, checkpointing, three steps, original shared_init and horizon3651.
Synchronized step times 9.719/2.470/3.003 seconds; peak allocated 42.418 GiB on rank0.
Timing includes transfer/tokenization, compute, backward, optimizer and scalar reductions, excludes
loader wait/checkpoint save/cohort diagnostics. Profiler and first-step warmup affect these values.
Historical ~21.7sec and 55.6GiB are the erroneous run's observations, not a controlled same-math benchmark.
The only tested chunk candidate was 32x64; no further tuning was necessary. See fix_v2_profile.json.

Actual dtypes: student image/text=bf16; router projection/logits, hard/proxy, ranking, frozen reference,
decoder and reconstruction=float32. Common init digest caf61198def1b78654d6b70ef16db9a3475ea81b3a16ab8a799989f574de2faf;
reference fingerprint 1428ea22801976c3358cf10905a64ba00bab7cadacda80e352016b3f8a2e1e9d.


## 修复版正式 500 步结果

代码提交：`063a49c666c29fd5a22a2baa320a2708d2166069`。本地与远端在启动前验证一致。
运行目录：`/root/SAID-token-v1/runs_salu/said_token_v1/fix_v2_step500_20260912`。从共同 shared_init 开始，未从旧 T1 恢复；保存 0/20/100/500。
正式训练仅此一次；前 20 步为同一运行的在线检查。四卡、每卡 256 对、全局 1024 对、LR horizon 3651、三组 warmup 200，全部学习率、温度和数据采样保持原配置。

### 数学与工程验收

IMPLEMENTATION: PASS。针对性反例、独立简单 reference、matched 对角线、单网格复用、分块和 checkpoint 梯度、真实两 rank DDP 同步均通过。
生产 step 1/5/20 的正配 score 梯度最大值均小于零，四卡梯度范数差为 0。已保存独立算术证据与数据流审计；旧运行不用于正确 T1 的方法优劣判断。
实际核心 dtype：学生输出 bf16；router 投影/logits、匹配 hard/soft、ranking、冻结参考、decoder、重建均为 FP32。
冻结参考最终指纹不变：`1428ea22801976c3358cf10905a64ba00bab7cadacda80e352016b3f8a2e1e9d`。
总墙钟 1022.27 秒；全部同步训练步骤平均 1.9895 秒；峰值显存（rank0 allocated）42.454 GiB。
25 步起的日志采样步骤中位数 1.9194 秒，均值 1.9814 秒。该子集不是所有步骤。
同步计时包含传输/tokenize/forward/backward/optimizer/标量通信，不含 DataLoader 等待、存档与固定 cohort；墙钟包含这些开销。短 profile 含 profiler 与启动开销，不把相对旧错误实现的比值包装成严格 benchmark。
Profile 剩余主要工作包括 FP32 矩阵乘、冻结参考塔与 NCCL gather/reduction；pairwise CPU 调度与重计算仍有成本。三步 rank0 汇总见 fix_v2_profile.json，kernel 累计时长可重叠，不能相加解释为墙钟。

### 训练轨迹

| step | Said（双向相加） | Rec | 正配 | 负配 | gap | soft gate mean | native CLS 跨图 cos | 同图 slot cos | 全图 slot cos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.169521 | 1.035190 | 0.719306 | 0.600695 | 0.118612 | 0.621367 | 0.429436 | 0.999624 | 0.810270 |
| 5 | 0.118974 | 1.036139 | 0.714269 | 0.562287 | 0.151982 | 0.591837 | 0.430283 | 0.999598 | 0.803569 |
| 20 | 0.113738 | 1.022913 | 0.716561 | 0.561396 | 0.155164 | 0.380537 | 0.420157 | 0.999624 | 0.805852 |
| 100 | 0.024897 | 0.488556 | 0.708587 | 0.430259 | 0.278328 | 0.000313 | 0.416873 | 0.999608 | 0.763862 |
| 200 | 0.004849 | 0.293833 | 0.686714 | 0.279615 | 0.407100 | 0.000540 | 0.309624 | 0.998044 | 0.661500 |
| 300 | 0.002484 | 0.233480 | 0.687403 | 0.203535 | 0.483868 | 0.000742 | 0.270108 | 0.994064 | 0.562762 |
| 400 | 0.001815 | 0.203456 | 0.692710 | 0.161434 | 0.531276 | 0.001073 | 0.237878 | 0.988677 | 0.498928 |
| 500 | 0.002121 | 0.183862 | 0.692304 | 0.135980 | 0.556324 | 0.001236 | 0.217353 | 0.985875 | 0.439491 |

训练曲线不能替代检索。soft gate 接近零说明低端饱和，日志中的 `soft_gate_saturated_fraction` 只统计 >0.999 的高端，不能据其为零声称“没有饱和”。top16 hard 选择仍按固定协议执行。

### 固定 64 图内容干预

| step | 正常重建误差 | 常数均值基线 | teacher bank R@1 | 预测方差 | 换文本 Δ | 换视觉内容 Δ | 同图 slot cos |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0498834 | 0.2922518 | 0.0312500 | 0.0000523 | 0.0007017 | -0.0007675 | 0.9996150 |
| 20 | 1.0326338 | 0.2922518 | 0.0156250 | 0.0000530 | 0.0009525 | -0.0006483 | 0.9996238 |
| 100 | 0.4344739 | 0.2922518 | 0.0156250 | 0.0001054 | 0.0073698 | 0.0023588 | 0.9995829 |
| 500 | 0.1606767 | 0.2922518 | 0.9687500 | 0.0006111 | 0.0676473 | 0.1295829 | 0.9839458 |

同一次干预保持 gate 固定，错视觉内容使用当前 mask 读取其他图 slots。诊断不调用训练 collectives。teacher-bank 指标是固定 cohort 诊断，不是独立语义基准。
修复版在固定 cohort 上出现内容依赖证据：step500 正常误差 0.16068，优于常数均值基线 0.29225；换文本增加约 0.06765，换视觉内容增加约 0.12958；teacher-bank R@1 为 62/64。结论仅限该固定输入诊断，不证明独立 Unsaid 语义或 U 对检索的增量贡献。

### 原生学生 CLS/EOS 检索

严格导出裸学生后只使用原生 encode_image/encode_text 的归一化输出；没有 teacher、decoder 或 fine-grained reranking。基线来自既有结果，没有重跑。COCO 和 Urban-1k 候选规模不同，数值不跨数据集比较。
| arm | COCO I2T R@1 | COCO T2I R@1 | Urban-1k I2T R@1 | Urban-1k T2I R@1 |
|---|---:|---:|---:|---:|
| Initial | 0.517000 | 0.326920 | 0.682000 | 0.528000 |
| S0@500 | 0.605800 | 0.412360 | 0.870000 | 0.842000 |
| C0@500 | 0.604200 | 0.407240 | 0.873000 | 0.834000 |
| C1@500 | 0.603800 | 0.412160 | 0.870000 | 0.837000 |
| T1_fix_v2@500 | 0.572800 | 0.378160 | 0.764000 | 0.713000 |

ShareGPT4V 固定 1K，与 Urban-1k 分开；所有 arm 的冻结 manifest 哈希一致。
| arm | variant | I2T R@1 | T2I R@1 |
|---|---|---:|---:|
| Initial | first_sentence | 0.544000 | 0.514000 |
| Initial | fixed_sparse | 0.746000 | 0.740000 |
| Initial | full_dense | 0.758000 | 0.776000 |
| S0@500 | first_sentence | 0.662000 | 0.635000 |
| S0@500 | fixed_sparse | 0.901000 | 0.879000 |
| S0@500 | full_dense | 0.965000 | 0.962000 |
| C0@500 | first_sentence | 0.667000 | 0.622000 |
| C0@500 | fixed_sparse | 0.903000 | 0.884000 |
| C0@500 | full_dense | 0.964000 | 0.962000 |
| C1@500 | first_sentence | 0.662000 | 0.634000 |
| C1@500 | fixed_sparse | 0.905000 | 0.883000 |
| C1@500 | full_dense | 0.969000 | 0.963000 |
| T1_fix_v2@500 | first_sentence | 0.622000 | 0.581000 |
| T1_fix_v2@500 | fixed_sparse | 0.834000 | 0.831000 |
| T1_fix_v2@500 | full_dense | 0.891000 | 0.883000 |

完整 R@1/R@5/R@10、原始全精度值、各数据集来源/manifest/checkpoint 哈希与 T1−S0/C0/C1 差值见 fix_v2_results.json。

冻结 COCO 晋升门：**FAIL**。使用原始 JSON 比较 I2T≥0.6058000000、T2I≥0.4123600000，且至少一项严格改善。没有以四舍五入值判定。

单 seed、500 步、不同架构/目标/算力预算的对照，不声称稳定普遍提升；数学修复通过不意味着检索一定更好。

### 可追溯性与未建立项

训练 checkpoint SHA256：`9262f53bc1729405061b9fcbab0ad14c9fbc97bfa9faf4fd4a260bdf44798c19`。
裸学生 SHA256：`3c45ed84f3a4729451210094dd719bddec7eddfec055f66b2da321907cb57b45`。
共同 init 文件 SHA256：`c1a4a2be1b212f38677f729a4f03160d788a6d6db79ca5021a43022af0f5cba8`；state digest 与文件哈希为不同定义。
报告提交 SHA 以普通提交后 git rev-parse HEAD / git ls-remote 的一致结果为准，在交付消息给出；不伪造自引用提交哈希。
UNSAID_SEMANTICS: NOT ESTABLISHED（未做独立语义基准）。
U reconstruction incremental contribution: NOT ESTABLISHED（未训练同结构 T0）。
T0、3 epoch、超参搜索、新损失/变体：NOT RUN（明确范围外）。旧错误 T1 的追加完整评估：NOT RUN（INVALID_FOR_T1_METHOD_COMPARISON）。
本轮完成后停止，不自动延长训练。
