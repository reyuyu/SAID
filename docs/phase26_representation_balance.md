# Phase 2.6：表征平衡与模态差距监控

本阶段研究 Caption 条件化后的 Said 表征是否更匹配当前文本的信息范围。
Phase 2.4 / 2.5 的定位审计、局部证据和 attention_delta 消融全部保留为历史分析。
生产默认仍为 `--said_feature_source residual`，使用末层 residual patch features。
单头 Router、q/k 归一化、温度、参数初始化、训练数据与 caption sampling 均未改变。

训练目标仍为 `L = lambda_global * L_global + lambda_said * L_said`，
其中 identifiable 模式 `L_said = 0.5 * (L_route + L_evidence)`。
默认两个 lambda 都为 1。没有新增 loss；Gap、L2M、RMG、PCA 不参与梯度和优化。

## 两种数据来源

在线 `pair_gap_full`、`pair_gap_said`、`balancing_gain`、`relative_balancing_gain`
保留到既有 JSONL，旧字段不删除。它们由当前 forward 中已有特征在 `no_grad` 下
以 fp32 累积，记录的是 **rank 0 本地训练 batch**，不是跨 rank 正式验证。
`balancing_gain = pair_gap_full - pair_gap_said`，相对增益除以
`max(pair_gap_full, 1e-8)`。训练损失、学习率、随机数流和数据采样不受影响。

正式 checkpoint 比较使用固定验证 probe，与 batch 曲线分开显示。
本轮复用已 Review 的 Phase 2.5 residual arm：initial、100、200、400、final（659 步）。
没有重跑完整 epoch。另跑 4 卡、每卡 batch 256、10-step smoke 验证新日志；
其 warmup、seed=25、损失和精度沿用生产设置，smoke 不冒充上述训练轨迹。

## 固定 manifest 与文本

数据仍为 676,415 条 no-SAM 子集；既有训练 Dataset 排除前 1,000 条后使用 675,415 条。
这不代表完整官方 1.246M SmartCLIP 训练复现。
probe 按源数据顺序，从前 1,000 条验证记录选取前 512 个唯一 image ID。
manifest 保存完整 dense caption、相对 image ID、dataset index、固定 order、训练式 prefix
构造信息、detail variants 与 level aliases、源 JSON SHA256、seed=26。
现有 manifest 与重新构造的规范不一致时拒绝覆盖，所有 checkpoint 使用同一个文件。

主 caption 按既有 ShareGPT4V 的 `replace("\n", " ").split(". ")` 构造，
使用独立 `random.Random(26 + dataset_index)` 选取 1..句数的均匀整数 prefix。
这是一次固定的验证 caption，不替换训练 caption、不改变训练 RNG。

detail ladder 对同一 dense caption 采用相同 `. ` 句界规则，去掉空片段并保持原句顺序：
第一句、前 ceil(25%)、ceil(50%)、ceil(75%)、100%，每级至少一句。
同一图像中重复 caption 只推理一次，通过 alias 映射保留五个命名 level 的同一 N 图像 cohort。
因此不会因去重而在某条曲线删除样本。该句界规则不是语言学 parser。
所有 variants 仅供离线诊断，训练代码不导入 probe 模块。

采用当前 LongCLIP tokenizer，context=248，`truncate=True` 与训练一致。
记录原始/使用 token 数与是否截断；“完整文本”指 ladder 的原始 dense caption，
并不保证模型读到了未截断的全部内容。文本长度和 detail 都只是信息覆盖代理。

## 高维表征与指标

每个 checkpoint 的 `t` 是该 checkpoint 当前文本编码器的归一化输出。
`z_full` 是标准 `encode_image(I)` 的全局表征；复用 patch extraction 返回的 global，
并在每个 checkpoint 首个图像 batch 与独立标准接口逐元素核对，不能 mean-pool patch 代替。
`z_said` 是当前 Router 的归一化输出。`z_base` 来自训练开始前 initial checkpoint 的
标准 global image representation，只计算一次并缓存，绑定 initial checkpoint 与 manifest SHA256。
评估逐个加载模型，训练过程中没有第二份常驻 frozen CLIP。

**Base 的图像表征固定，但每个 checkpoint 都与当前的 t 配对。**
所以 Base Gap / L2M / RMG 可因文本编码器漂移而变化；它不是“两个端点都冻结”的曲线。
标准 global 分支也在训练，因此不能把相对 Full 的比较理解为单独 Router 的因果效应。

所有正式数值由 L2-normalized **512D** embeddings 在 NumPy float64 中计算：

```text
d(v,t) = 1 - cosine(v,t)
Pair Gap = P = mean_i d(v_i,t_i)
L2M = ||mean_i(v_i) - mean_i(t_i)||_2
W_v = mean_{i != j} d(v_i,v_j)
W_t = mean_{i != j} d(t_i,t_j)
RMG = P / (P + 0.5*(W_v + W_t))
```

模态内均值排除自配对，按全部有序 i!=j 项计算；通过向量和的平方范数精确求和。
单元测试另用小型手算例子，实测 artifact 审核还用显式 off-diagonal 距离矩阵独立核验。
N 必须至少 2，零向量/非有限输入拒绝计算。仅裁剪浮点余弦超界误差；不平滑或强制趋势。
若 RMG 分母 <= 1e-12，输出 JSON null 与 `undefined_zero_denominator`，不把完全塌缩
伪装成优秀的 0 Gap 比率。正常值严格使用上述公式。

RMG 参考 [Two Effects, One Trigger（ICLR 2025，式 1）](https://proceedings.iclr.cc/paper_files/paper/2025/file/4572bc2f514e627914cbe60d0398a2d1-Paper-Conference.pdf)
及[作者官方 compute_rmg](https://github.com/lmb-freiburg/two-effects-one-trigger/blob/main/analysis/gap_precompute.py)。
官方代码将 cosine similarity 从 [-1,1] 映射到 [0,1] 后取距离，即本实现距离的 1/2；
对正常的 P、W 同比缩放会抵消，RMG 相同。官方对非正 P 替换为 1e-3；
本实现不采用该替换，零分母明确未定义。官方支持一图多 caption 的配对映射，
本 probe 为每图一个固定主 caption，每个 detail level 也为每图一个 caption。
不将不同 cohort 或不同文本配对的 RMG 直接当作可比绝对分数。

## Caption 条件化对照

固定 `j=(i+1)%N`，不随机 shuffle。Own 与 Shuffle 两种 Router 输出都与目标 `t_i` 比较：
`conditioning_gap_margin = said_shuffle_gap - said_own_gap`。
正值支持 caption conditioning sanity check，不表示 grounding accuracy。
所有样本和所有指定 checkpoint 都保留；不选择性删除不支持假设的结果。

## PCA 与前端

每个 checkpoint 对 `[Z_base; Z_full; Z_said; T]` **联合 fit 一次** PCA；
三个 panel 使用相同 basis、相同中心化均值和相同 x/y limits。
PCA 符号通过最大绝对 loading 取正固定；协方差椭圆用二维高斯 95% contour
（chi-square df=2 = 5.991464547），不是均值置信区间。固定连接前 30 对。
范围包含全部点和椭圆，避免裁切；不同 checkpoint 各自拟合，不比较跨阶段二维绝对坐标。
PCA 仅定性展示，不参与正式距离计算。

Streamlit 默认首页为“表征平衡监控”，另有“训练状态”和四个历史页面。
全部新页面标题、图例、指标解释、空数据与错误提示以中文呈现；历史页面同步中文化。
文件使用 UTF-8，JSON `ensure_ascii=False`。
浏览器图表字体按 Microsoft YaHei、PingFang SC、Noto Sans CJK SC、Source Han Sans SC、
Arial Unicode MS、sans-serif 回退，不要求服务器安装中文字体。
每行最多三张指标卡，图表随容器宽度伸缩，小屏布局自动换行。
前端只读 JSON，不加载 checkpoint、NPZ embeddings 或 GPU 模型；页面刷新不推理。
COCO 缺失时显示“当前 checkpoint 尚未运行 COCO Retrieval 评估。”。
历史库的原始 JSON/诊断 key 与第三方 Streamlit 自带工具按钮保留原始标识。

## 复现实验

在仓库根目录、已有 said-smartclip 环境运行，数据根目录通过环境变量提供：

```bash
bash train/run_representation_gap_smoke.sh

python -m eval.salu.representation_balance_eval \
  --dataset_json "$SHARE4V_DATA_ROOT/$SHARE4V_JSON" \
  --image_root "$SHARE4V_DATA_ROOT" \
  --checkpoints_root runs_salu/phase25/residual \
  --batch_log runs_salu/phase26_gap_smoke/salu_log.jsonl \
  --coco_artifact outputs/local_evidence_router/coco.json \
  --output outputs/representation_balance

python -m pytest -q tests
streamlit run tools/said_dashboard/app.py --server.address 127.0.0.1
```

smoke 脚本拒绝覆盖已有日志。导出器重用且校验现有 manifest/base cache；
对同一个规范重新导出 checkpoint 会更新该目录的诊断文件。
COCO 仅在 checkpoint SHA256 相符时复用 reviewed residual final 结果，其他阶段不填造。

输出根目录包括 manifest.json、summary.json、base_cache.npz、batch_history.json；
五个阶段各有 embeddings.npz、gap_metrics.json、caption_detail.json、pca.json。
新输出独立放在 `outputs/representation_balance/`，不修改历史实验。
输出、checkpoint、训练日志、NPZ、PNG 均不入 Git。

本阶段完成后等待 Review，不进入 Prototype / Unsaid。
