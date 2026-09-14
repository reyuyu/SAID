# SAID 数据与权重

GitHub 同步代码、说明和小型证据；训练数据、模型和缓存独立存储。两个实验的文件必须按版本、更新数及 SHA256 核对，不能只依据相同的 `step003651.pt` 文件名。

## 检查点身份

| 版本 / 类型 | 更新数 | SHA256 |
|---|---:|---|
| 共同初始化 cvssl_initial.pt | 0 | `c1a4a2be1b212f38677f729a4f03160d788a6d6db79ca5021a43022af0f5cba8` |
| Clean 完整 checkpoint | 3651 | `085f0e019f01e5318e81bafe4d676f925d9b4b03497e15765e0f1fad9a97ba32` |
| Clean bare student | 3651 | `aa8aab9ef7184fb050cd2596811424e99752005b1651e00f998620cbbdc9f62a` |
| Full 完整 checkpoint | 3651 | `448dd0b94b8b1a26ffcfa4b3e7f8f44ecd069d886977ce79454f4844dcd41773` |
| Full bare student（原导出） | 3651 | `682b89f6efa2c59d4ecbfa7d4f44c8ac6ed926476355d17a40a56a76fd15ad19` |
| Full bare student（本地完整权重重新导出，扩展评测使用） | 3651 | `1d0dbc650670e809ae2ea2c5fcc75221cd626ac954f071b69ddffaa103e75f79` |
| Full 完整 checkpoint | 2000 | `bfaeeb4a54c1c76702831472e351aab3ab2b0ae9e231d87bd5fea7544d20b5da` |
| Full bare student | 2000 | `54f5c8b601c237e883a79cb82cb85865e3e0ff27e7399a189eb95c21361faf15` |
| Full Urban 适配导出 | 2000 | `9225a1b912f2ac51fddb74a4f5e7bce5eac2a323d68cbb842c583003db1e9c29` |

裸学生约612 MB，完整含优化器检查点约1.84 GB；不同导出容器有不同文件哈希。原始 metadata 和检索结果均保留各自实际文件身份。

## 训练与评测数据

共享训练输入为 ShareGPT4V 的 `share-captioner_coco_lcs_sam_1246k_1107.json` 与所引用图片。跳过前1000行后训练样本数1,245,901，原随机前K句、后缀排除末句规则不变。完整输入详情见 [训练数据清单](s0_dualmask_masked_3epoch/manifests/training_data.json)；初始化、base CLIP 与标注哈希见 [assets.json](s0_dualmask_masked_3epoch/manifests/assets.json)。

| 协议 | 图片 / 文本 | 对齐清单标识 |
|---|---:|---|
| COCO canonical | 5000 / 25000 | annotations SHA `afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9` |
| Urban-1k | 1000 / 1000 | pair-caption SHA `3b0b2a3b743ed6011fccc72f8a9777bec35b44cadf2483f707fc5ef317da996c` |
| docci | 5000 / 5000 | manifest SHA `e852a96b4efb9fa6585fd70b2686cb4e456409b1144c4a0e3c7bd8a24a36cb11` |
| dci | 7805 / 7805 | manifest SHA `14530fb8bf3c7b4a75bb451412d4562c548f9ca1bdbe1fdbf39d6ee30fb4de24` |
| long_dci（重建版） | 7602 / 7602 | manifest SHA `8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b` |
| flickr_test1k | 1000 / 5000 | manifest SHA `113dbc616ca66db9400107ed4b97b56d33adae3c18608225902d97ee943600dc` |

数据放在机器外部目录，迁移步骤见 [TRANSFER_AND_UPLOAD.md](s0_dualmask_masked_3epoch/TRANSFER_AND_UPLOAD.md)。训练图片仅路径存在并不能证明图片字节一致；跨服务器需保留原文件并进行额外校验。

## 私人备份状态

维护者已在科大云盘 `CCCLIP` 保存部分代码包、两个版本最终权重及 Clean step1000、部分评测资产。该目录是私人备份，不是公开下载端点；未发布可下载权重附件时，外部读者需先向维护者取得对应文件。

2026-09-14 的上传记录仍显示完整 ShareGPT4V 训练数据、扩展检索全量数据与全部中间检查点**尚未全部同步**；旧的约1.66 GB/3000文件训练子集不能充当正式训练集。GitHub 本次同步不改变这个备份状态。
