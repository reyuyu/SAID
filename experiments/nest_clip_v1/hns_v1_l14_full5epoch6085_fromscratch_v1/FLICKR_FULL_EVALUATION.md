# Flickr30k-Full 补充评测

使用冻结的官方 token 协议：31,783 张图像、158,915 条 caption，manifest SHA256 为 `bfed72a299e4d8cbfef817c13c836cd5aa189e18dd2fd29b632f5808e8dd12bc`。评测加载严格导出的裸学生，使用 ViT-L/14、248-token tokenizer、原始图像预处理、FP32 编码、L2 归一化和分块全候选池检索；没有使用 mask、rerank、ensemble 或 TTA。

| 节点 | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---:|---:|---:|---:|---:|---:|
| Full5@4868 | 64.710694 | 84.611270 | 90.353334 | 44.405500 | 67.085549 | 75.349086 |
| Full5@6085 | 64.701255 | 84.611270 | 90.375358 | 44.409275 | 67.100022 | 75.360413 |

两次评测均报告 `status=COMPLETED`，模型加载前后状态摘要一致。原始结果和执行收据保存在服务器 runtime 的 `evaluation_final/flickr_full/step4868/RESULT.json` 与 `step6085/RESULT.json`。
