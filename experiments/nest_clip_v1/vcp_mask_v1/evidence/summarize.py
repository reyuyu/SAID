#!/usr/bin/env python3
"""Build the compact VCP-Mask/TI-noInc 500-step result bundle."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EXP = REPO / "experiments/nest_clip_v1/vcp_mask_v1"
FORMAL = Path("/root/lk_projects/SAID-nest-clip-v1/vcp_mask_v1/formal")
REFERENCE_PATH = REPO / "experiments/nest_clip_v1/jointmask_fast_v1/FORMAL500_RESULTS.json"
REFERENCE_FORMAL = Path("/root/lk_projects/SAID-nest-clip-v1/jointmask_fast_v1/formal")
NEW_GROUPS = ("VCP-Mask", "TI-noInc")
REFERENCE_GROUPS = ("T-fast", "TI-fast")
ALL_GROUPS = REFERENCE_GROUPS + NEW_GROUPS
DATASETS = ("COCO", "Urban-1k", "Flickr30k-test1k", "DOCCI")
DIRECTIONS = ("I2T", "T2I")
RECALLS = ("R@1", "R@5", "R@10")
KEY_STEPS = (1, 100, 200, 201, 300, 400, 500)
TRAIN_FIELDS = (
    "loss", "common_loss", "F_i2t", "F_t2i", "O_i2t", "O_t2i", "E_i2t", "E_t2i",
    "F_sparse", "O_sparse", "E_sparse", "F_keep_ratio", "O_keep_ratio", "E_keep_ratio",
    "F_positive_keep_ratio", "O_positive_keep_ratio", "E_positive_keep_ratio",
    "F_negative_keep_ratio", "O_negative_keep_ratio", "E_negative_keep_ratio",
    "F_all_open", "F_all_closed", "O_all_open", "O_all_closed", "E_all_open", "E_all_closed",
    "inc", "inc_weight", "hard_inclusion_violation", "oe_iou", "valid_global", "nonfinite",
    "F_delta_abs_mean", "O_delta_abs_mean", "E_delta_abs_mean", "q_minus_w_norm", "w_norm",
    "F_pooling_entropy", "O_pooling_entropy", "E_pooling_entropy",
    "F_pooling_l1_from_text_only", "O_pooling_l1_from_text_only", "E_pooling_l1_from_text_only",
    "F_hard_mask_switch_fraction", "O_hard_mask_switch_fraction", "E_hard_mask_switch_fraction",
)


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_recall(block):
    return {f"R@{key[1:]}" if key.startswith("R") and not key.startswith("R@") else key: value
            for key, value in block.items()}


def evaluation(group: str):
    root = FORMAL / group
    coco = load(root / "coco_native.json")
    urban = load(root / "urban_native.json")
    flickr = load(root / "flickr_test1k/flickr_test1k.json")
    docci = load(root / "docci/docci.json")
    metrics = {
        "COCO": {
            "I2T": {"R@1": coco["image2text_R1"], "R@5": coco["image2text_R5"],
                    "R@10": coco["image2text_R10"]},
            "T2I": {"R@1": coco["text2image_R1"], "R@5": coco["text2image_R5"],
                    "R@10": coco["text2image_R10"]},
        },
        "Urban-1k": {"I2T": normalize_recall(urban["image2text"]),
                     "T2I": normalize_recall(urban["text2image"])},
    }
    for name, raw in (("Flickr30k-test1k", flickr), ("DOCCI", docci)):
        metrics[name] = {direction: {} for direction in DIRECTIONS}
        for recall, pair in raw["metrics"].items():
            for direction in DIRECTIONS:
                metrics[name][direction][recall] = pair[direction]
    metadata = {
        "COCO": {"n_images": 5000, "n_captions": 25000, "similarity_chunk": 512,
                 "batch_size": 64, "checkpoint_sha256": coco["checkpoint_sha256"],
                 "native_only": coco["native_only"]},
        "Urban-1k": {k: urban[k] for k in urban if k not in ("root", "image2text", "text2image")},
        "Flickr30k-test1k": {k: flickr[k] for k in flickr if k != "metrics"},
        "DOCCI": {k: docci[k] for k in docci if k != "metrics"},
    }
    sources = {
        "COCO": root / "coco_native.json",
        "Urban-1k": root / "urban_native.json",
        "Flickr30k-test1k": root / "flickr_test1k/flickr_test1k.json",
        "DOCCI": root / "docci/docci.json",
    }
    return metrics, metadata, sources


def metric_value(row, key):
    if key == "common_loss":
        return row["loss"] - row["inc_weight"] * row["inc"]
    return row.get(key)


def compact_step(row):
    out = {key: metric_value(row, key) for key in TRAIN_FIELDS if metric_value(row, key) is not None}
    out.update({key: row[key] for key in ("F_candidates", "O_candidates", "E_candidates")})
    return out


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    weight = index - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def stream_signatures(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    digest_keys = ("sample_id_sha256", "full_view_sha256", "local_views_sha256",
                   "split_sha256", "fixed_first_reference_stream_sha256")
    return [[{"rank": rank["rank"], "stream_sha256": rank["stream_sha256"],
              **{key: rank["sampling"][key] for key in digest_keys}}
             for rank in row["rank_health"]] for row in rows]


def training(group: str):
    root = FORMAL / group
    rows = [json.loads(line) for line in (root / "steps.jsonl").read_text().splitlines() if line.strip()]
    by_step = {row["step"]: row for row in rows}
    last50 = rows[-50:]
    means = {}
    for key in TRAIN_FIELDS:
        values = [metric_value(row, key) for row in last50]
        values = [value for value in values if value is not None]
        if values:
            means[key] = statistics.fmean(values)
    max_rank_step = [max(rank["seconds"] for rank in row["rank_health"]) for row in rows]
    regular = max_rank_step[5:]
    acceptance = load(root / "acceptance.json")
    config = load(root / "config.json")
    export = load(root / "export-check.json")
    stream_rows = stream_signatures(root / "steps.jsonl")
    return {
        "config": config,
        "acceptance": acceptance,
        "export_check": export,
        "checkpoint_sha256": {name: sha256(root / name) for name in
                              ("step000000.pt", "step000100.pt", "step000200.pt",
                               "step000300.pt", "step000400.pt", "step000500.pt")},
        "key_steps": {str(step): compact_step(by_step[step]) for step in KEY_STEPS},
        "last50_mean": means,
        "timing": {
            "training_seconds": max(rank["seconds"] for rank in acceptance["ranks"]),
            "startup_step_seconds": max_rank_step[0],
            "regular_steps_definition": "steps 6-500; per-step maximum over four ranks",
            "regular_step_mean_seconds": statistics.fmean(regular),
            "regular_step_median_seconds": statistics.median(regular),
            "regular_step_p90_seconds": percentile(regular, 0.90),
            "regular_step_p95_seconds": percentile(regular, 0.95),
            "regular_step_max_seconds": max(regular),
        },
        "integrity": {
            "records": len(rows),
            "steps_continuous": [row["step"] for row in rows] == list(range(1, 501)),
            "nonfinite_sum": sum(row["nonfinite"] for row in rows),
            "all_gradients_finite": all(all(rank["gradients_finite"] for rank in row["rank_health"])
                                        for row in rows),
            "fallback_steps": sum(row["valid_global"] < 2 for row in rows),
            "duplicate_id_steps": sum(bool(row["duplicate_image_ids"]) for row in rows),
            "valid_global_min": min(row["valid_global"] for row in rows),
            "valid_global_max": max(row["valid_global"] for row in rows),
            "F_candidates_min": min(row["F_candidates"] for row in rows),
            "F_candidates_max": max(row["F_candidates"] for row in rows),
            "O_candidates_min": min(row["O_candidates"] for row in rows),
            "O_candidates_max": max(row["O_candidates"] for row in rows),
            "E_candidates_min": min(row["E_candidates"] for row in rows),
            "E_candidates_max": max(row["E_candidates"] for row in rows),
        },
        "_stream_rows": stream_rows,
    }


def comparison(metrics, left, right):
    delta = {}
    flat = []
    r1 = []
    wins = ties = losses = 0
    for dataset in DATASETS:
        delta[dataset] = {}
        for direction in DIRECTIONS:
            delta[dataset][direction] = {}
            for recall in RECALLS:
                value = 100 * (metrics[left][dataset][direction][recall]
                               - metrics[right][dataset][direction][recall])
                delta[dataset][direction][recall] = value
                flat.append(value)
                if recall == "R@1":
                    r1.append(value)
                if value > 1e-12:
                    wins += 1
                elif value < -1e-12:
                    losses += 1
                else:
                    ties += 1
    return {"delta_pp": delta, "mean_delta_pp_all_24": statistics.fmean(flat),
            "mean_delta_pp_r1_8": statistics.fmean(r1), "wins_ties_losses": [wins, ties, losses]}


def j_long(metrics):
    return statistics.fmean((
        metrics["Urban-1k"]["I2T"]["R@1"], metrics["Urban-1k"]["T2I"]["R@1"],
        metrics["DOCCI"]["I2T"]["R@1"], metrics["DOCCI"]["T2I"]["R@1"],
    ))


def fmt_pct(value):
    return f"{100 * value:.2f}"


def fmt_pp(value):
    return f"{value:+.2f}"


def main():
    report_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    reference = load(REFERENCE_PATH)
    reference_sha = sha256(REFERENCE_PATH)
    metrics = {group: {dataset: reference["metrics"][group][dataset] for dataset in DATASETS}
               for group in REFERENCE_GROUPS}
    metadata = {group: {dataset: reference["evaluation_metadata"][group][dataset]
                        for dataset in DATASETS} for group in REFERENCE_GROUPS}
    groups = {group: reference["groups"][group] for group in REFERENCE_GROUPS}
    source_files = {}
    for group in NEW_GROUPS:
        group_metrics, group_metadata, sources = evaluation(group)
        metrics[group] = group_metrics
        metadata[group] = group_metadata
        source_files[group] = sources
        groups[group] = training(group)

    streams_equal = groups["VCP-Mask"]["_stream_rows"] == groups["TI-noInc"]["_stream_rows"]
    reference_streams = stream_signatures(REFERENCE_FORMAL / "TI-fast/steps.jsonl")
    streams_equal_reference = groups["VCP-Mask"]["_stream_rows"] == reference_streams
    for group in NEW_GROUPS:
        del groups[group]["_stream_rows"]

    comparisons = {
        "TI-noInc_minus_TI-fast": comparison(metrics, "TI-noInc", "TI-fast"),
        "VCP-Mask_minus_TI-fast": comparison(metrics, "VCP-Mask", "TI-fast"),
        "VCP-Mask_minus_T-fast": comparison(metrics, "VCP-Mask", "T-fast"),
    }
    j_values = {group: j_long(metrics[group]) for group in ALL_GROUPS}
    j_deltas = {
        "TI-noInc_minus_TI-fast_pp": 100 * (j_values["TI-noInc"] - j_values["TI-fast"]),
        "VCP-Mask_minus_TI-fast_pp": 100 * (j_values["VCP-Mask"] - j_values["TI-fast"]),
        "VCP-Mask_minus_T-fast_pp": 100 * (j_values["VCP-Mask"] - j_values["T-fast"]),
    }
    speed = {
        group: load(EXP / "evidence" / f"probe-{group}-clean-acceptance.json")
        for group in NEW_GROUPS
    }

    result = {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "branch": "codex/nest-vcp-mask-v1",
        "report_source_head": report_head,
        "training_git_head": groups["VCP-Mask"]["config"]["git_head"],
        "reference_commit": "6bafa8a009af4ffee1100027d5171d78dbeb96d3",
        "reference_results_path": str(REFERENCE_PATH.relative_to(REPO)),
        "reference_results_sha256": reference_sha,
        "initial_checkpoint_sha256": "54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6",
        "protocol": {
            "training": {"world_size": 4, "batch_per_rank": 256, "global_batch": 1024,
                         "accumulation": 1, "seed": 0, "updates": 500,
                         "scheduler_horizon": 3651, "training_records": 1245901,
                         "checkpoint_encoders": True, "precision": "FP32 params; BF16 encoders; FP32 gate/score/loss"},
            "evaluation": {"batch_size": 64, "native_student_only": True,
                           "similarity": "normalized image/text embeddings inner product",
                           "datasets": list(DATASETS), "excluded": ["DCI", "long-DCI"]},
        },
        "tests": {
            "unit_regression": {"passed": 48, "failed": 0,
                                "log": "evidence/unit-tests-after-runner-fixes.console.txt"},
            "vcp_ddp_reference": load(EXP / "evidence/vcp-ddp-validation.json"),
            "smoke": load(EXP / "evidence/smoke-audit.json"),
        },
        "speed_gate": speed,
        "stream_digests_equal_new_groups_all_steps_all_ranks": streams_equal,
        "stream_digests_equal_reference_TI_fast_all_steps_all_ranks": streams_equal_reference,
        "groups": groups,
        "metrics": metrics,
        "evaluation_metadata": metadata,
        "j_long": {"definition": "mean Urban/DOCCI I2T/T2I R@1", "values": j_values,
                   "deltas_pp": j_deltas},
        "comparisons": comparisons,
        "recommendation": {
            "retain": ["TI-fast"],
            "do_not_combine_without_new_control": ["VCP-Mask", "TI-noInc"],
            "reason": "Neither new group improved J_long or the overall native-retrieval profile over TI-fast at 500 updates, seed 0."
        },
        "limitations": [
            "single seed and a single 500-update stopping point",
            "benchmarks were used during exploration and are not an independent blind test",
            "native evaluation removes the training-only mask/adapter, so results measure student representation changes",
            "no DCI or long-DCI evaluation was run by request",
        ],
    }

    raw_root = EXP / "evidence/native_results"
    for group, sources in source_files.items():
        destination = raw_root / group
        destination.mkdir(parents=True, exist_ok=True)
        for dataset, source in sources.items():
            shutil.copy2(source, destination / f"{dataset.lower().replace('-', '_')}.json")
        shutil.copy2(FORMAL / group / "export-check.json", destination / "export-check.json")
        shutil.copy2(FORMAL / group / "config.json", destination / "training-config.json")
        shutil.copy2(FORMAL / group / "acceptance.json", destination / "training-acceptance.json")

    result_path = EXP / "FORMAL500_RESULTS.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    vcp_vs_ti = comparisons["VCP-Mask_minus_TI-fast"]
    noinc_vs_ti = comparisons["TI-noInc_minus_TI-fast"]
    lines = [
        "# NEST VCP-Mask / TI-noInc：500步实验报告",
        "",
        "日期：2026-09-30 UTC",
        "",
        f"正式训练代码：`{result['training_git_head']}`，分支 `codex/nest-vcp-mask-v1`。"
        f"对照来自提交 `6bafa8a009af4ffee1100027d5171d78dbeb96d3` 的原始结果JSON"
        f"（SHA256 `{reference_sha}`）。",
        "",
        "两组均从共同step-0独立启动，初始化SHA256为 "
        "`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`。"
        "训练使用单机4张A100 80GB、每rank batch 256、全局batch 1024、seed 0、"
        "scheduler horizon 3651，并在500次同步更新停止。",
        "",
        "## 结论",
        "",
        f"本轮两个新增组都通过3秒资源门、5步四卡smoke和500步正式训练，但都没有超过TI-fast。"
        f"预先固定的 `J_long` 中，TI-fast为 **{fmt_pct(j_values['TI-fast'])}%**，"
        f"TI-noInc为 **{fmt_pct(j_values['TI-noInc'])}%**（{fmt_pp(j_deltas['TI-noInc_minus_TI-fast_pp'])} pp），"
        f"VCP-Mask为 **{fmt_pct(j_values['VCP-Mask'])}%**（{fmt_pp(j_deltas['VCP-Mask_minus_TI-fast_pp'])} pp）。",
        "",
        f"TI-noInc相对TI-fast在24个Recall指标中为"
        f"{noinc_vs_ti['wins_ties_losses'][0]}胜/{noinc_vs_ti['wins_ties_losses'][1]}平/"
        f"{noinc_vs_ti['wins_ties_losses'][2]}负，24项平均变化"
        f"{noinc_vs_ti['mean_delta_pp_all_24']:+.3f} pp。关闭包含损失没有带来整体收益，"
        "尤其Flickr30k和DOCCI的R@1整体更弱；本轮证据支持保留TI-fast的包含项。",
        "",
        f"VCP-Mask相对TI-fast为{vcp_vs_ti['wins_ties_losses'][0]}胜/"
        f"{vcp_vs_ti['wins_ties_losses'][1]}平/{vcp_vs_ti['wins_ties_losses'][2]}负，"
        f"24项平均变化{vcp_vs_ti['mean_delta_pp_all_24']:+.3f} pp。它相对T-fast的"
        f"`J_long`也为{fmt_pp(j_deltas['VCP-Mask_minus_T-fast_pp'])} pp，结果接近但没有形成"
        "可解释的长文本检索优势。因此当前建议继续保留TI-fast，不启动VCP+noInc组合。",
        "",
        "这些差异来自单个seed和一个500步停止点，且基准已用于机制探索，不代表统计显著、盲测结论或最终最佳模型。",
        "",
        "## 原生检索结果",
        "",
        "数值为百分比；两个差值列为百分点，分别是TI-noInc−TI-fast与VCP-Mask−TI-fast。"
        "评测只使用裸学生的归一化图文向量内积，不使用mask、adapter、融合或rerank。",
        "",
        "| 数据集 | 方向 | 指标 | T-fast | TI-fast | TI-noInc | VCP-Mask | noInc−TI | VCP−TI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for direction in DIRECTIONS:
            for recall in RECALLS:
                lines.append("| " + " | ".join((dataset, direction, recall,
                    fmt_pct(metrics["T-fast"][dataset][direction][recall]),
                    fmt_pct(metrics["TI-fast"][dataset][direction][recall]),
                    fmt_pct(metrics["TI-noInc"][dataset][direction][recall]),
                    fmt_pct(metrics["VCP-Mask"][dataset][direction][recall]),
                    fmt_pp(noinc_vs_ti["delta_pp"][dataset][direction][recall]),
                    fmt_pp(vcp_vs_ti["delta_pp"][dataset][direction][recall]))) + " |")

    lines += [
        "",
        "### 固定长文本汇总",
        "",
        "`J_long = mean(Urban I2T R@1, Urban T2I R@1, DOCCI I2T R@1, DOCCI T2I R@1)`。",
        "",
        "| 组别 | J_long | 相对TI-fast |",
        "|---|---:|---:|",
    ]
    for group in ALL_GROUPS:
        lines.append(f"| {group} | {fmt_pct(j_values[group])}% | "
                     f"{fmt_pp(100 * (j_values[group] - j_values['TI-fast']))} pp |")

    lines += [
        "",
        "## 训练机制",
        "",
        "下表为最后50步均值。`common loss = total loss − inc_weight × inc`，因此可以排除"
        "A3包含正则对总loss的直接加和。O/E沿用实现字段，对应提示词中的P/R视图。",
        "",
        "| 组别 | common loss | inc/weight | hard违例 | O/E IoU | F/O/E keep | F全开/全关 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ALL_GROUPS:
        m = groups[group]["last50_mean"]
        lines.append(f"| {group} | {m['common_loss']:.4f} | {m['inc']:.4f}/{m['inc_weight']:.3f} | "
                     f"{m['hard_inclusion_violation']:.4f} | {m['oe_iou']:.4f} | "
                     f"{m['F_keep_ratio']:.4f}/{m['O_keep_ratio']:.4f}/{m['E_keep_ratio']:.4f} | "
                     f"{m['F_all_open']:.4f}/{m['F_all_closed']:.4f} |")

    vcp_m = groups["VCP-Mask"]["last50_mean"]
    ti_m = groups["TI-fast"]["last50_mean"]
    noinc_m = groups["TI-noInc"]["last50_mean"]
    lines += [
        "",
        f"关闭包含项后，TI-noInc最后50步hard包含违例为{noinc_m['hard_inclusion_violation']:.4f}，"
        f"TI-fast为{ti_m['hard_inclusion_violation']:.4f}。这说明本轮关闭正则没有通过其他损失"
        "自然恢复同等的包含约束；原生检索也没有改善。",
        "",
        f"VCP-Mask最后50步的 `||q-w||` 为{vcp_m['q_minus_w_norm']:.4f}，"
        f"`||w||` 为{vcp_m['w_norm']:.4f}；F/O/E相对纯文本池化的hard mask切换比例为"
        f"{vcp_m['F_hard_mask_switch_fraction']:.4f}/"
        f"{vcp_m['O_hard_mask_switch_fraction']:.4f}/"
        f"{vcp_m['E_hard_mask_switch_fraction']:.4f}。视觉查询确实改变了池化和mask，"
        "但这种机制变化没有转化为更好的裸学生原生检索。",
        "",
        "## 资源门、训练完整性与导出",
        "",
        "速度门使用真实DataLoader和完整4×256同步更新：预热5步后连续测30步，每步取四rank最大值。",
        "",
        "| 组别 | 速度门均值/中位/P95/最大 s | 峰值allocated GiB | 正式训练分钟 | 正式常规步均值/P95/最大 s | rank参数最大差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in NEW_GROUPS:
        gate = speed[group]["speed_gate"]
        acc = groups[group]["acceptance"]["ranks"]
        timing = groups[group]["timing"]
        lines.append(f"| {group} | {gate['mean_seconds']:.3f}/{gate['median_seconds']:.3f}/"
                     f"{gate['p95_seconds']:.3f}/{gate['max_seconds']:.3f} | "
                     f"{max(x['peak_allocated_gib'] for x in acc):.2f} | "
                     f"{timing['training_seconds']/60:.2f} | {timing['regular_step_mean_seconds']:.3f}/"
                     f"{timing['regular_step_p95_seconds']:.3f}/{timing['regular_step_max_seconds']:.3f} | "
                     f"{max(x['max_parameter_difference_from_rank0'] for x in acc):.1f} |")

    lines += [
        "",
        "两组均有连续且无重复的step 1–500日志；四个rank各完成500次更新；loss和梯度均有限；"
        "F候选始终为1024，O/E全局有效候选为1022–1024；没有F-only回退或重复ID步骤；"
        "最终参数相对rank0最大差异为0，最终NCCL all-reduce检查通过。两组所有step、所有rank的"
        f"样本/文本/token流摘要在两个新增组之间完全一致：`{str(streams_equal).lower()}`，且与既有TI-fast参考的"
        f"全部500步、全部rank完全一致：`{str(streams_equal_reference).lower()}`。",
        "",
        "VCP-Mask与TI-noInc均保存step0/100/200/300/400/500。严格导出检查显示optimizer step为500，"
        "完整训练模型与裸学生的native image/text embedding最大绝对误差均为0。",
        "",
        "| 组别 | step500训练checkpoint SHA256 | 裸学生SHA256 |",
        "|---|---|---|",
    ]
    for group in NEW_GROUPS:
        lines.append(f"| {group} | `{groups[group]['export_check']['checkpoint_sha256']}` | "
                     f"`{groups[group]['export_check']['bare_sha256']}` |")

    lines += [
        "",
        "必要测试共48项通过。两rank可导gather参考覆盖全有效、单rank零有效、全局V=1、V=0和尾批；"
        "loss/梯度及真实AdamW一步更新均在既定容差内。near-zero bias参数出现约2e-3的AdamW差异，"
        "已逐参数记录，没有通过删除梯度或整体放宽阈值掩盖。两组5步真实四卡smoke均通过。",
        "",
        "资源探针开发过程中曾出现更新完成后的DataLoader/NCCL清理竞态。该问题发生在探针退出阶段，"
        "未复用其权重；通过让短运行DataLoader自然耗尽并显式等待collective后，最终两次干净速度门均通过。"
        "早期失败日志保留在evidence目录。",
        "",
        "## 评测范围与产物",
        "",
        "COCO为5000图/25000文本、similarity chunk 512；Urban-1k为1000对；"
        "Flickr30k test1K为1000图/5000文本；DOCCI为5000对。图像batch为64。"
        "按要求没有运行DCI或long-DCI。",
        "",
        "机器可读汇总为 `FORMAL500_RESULTS.json`。新增组的原始评测JSON、运行配置、训练验收和导出核验"
        "位于 `evidence/native_results/`；测试、速度门、smoke、正式训练和评测控制台记录位于"
        "`evidence/`。权重、数据、缓存及大型trace只保留在服务器，没有加入Git。",
        "",
    ]
    (EXP / "FORMAL500_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"results": str(result_path), "report": str(EXP / "FORMAL500_REPORT.md"),
                      "streams_equal": streams_equal, "streams_equal_reference": streams_equal_reference,
                      "j_long": j_values,
                      "j_long_deltas_pp": j_deltas}, indent=2))


if __name__ == "__main__":
    main()
