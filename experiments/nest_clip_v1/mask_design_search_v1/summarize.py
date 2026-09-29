#!/usr/bin/env python3
"""Assemble the JointInput resource-gate result and report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EVIDENCE = HERE / 'evidence'
REFERENCE = REPO / 'experiments/nest_clip_v1/jointmask_fast_v1/FORMAL500_RESULTS.json'


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def j_long(metrics):
    values = [metrics['Urban-1k']['I2T']['R@1'], metrics['Urban-1k']['T2I']['R@1'],
              metrics['DOCCI']['I2T']['R@1'], metrics['DOCCI']['T2I']['R@1']]
    return sum(values) / len(values)


def main():
    reference = load(REFERENCE)
    preflight = load(EVIDENCE / 'preflight.json')
    actual = load(EVIDENCE / 'actual-validation.json')
    ddp = load(EVIDENCE / 'ddp-validation.json')
    selected_files = {
        'C1 CLS-All': 'cls-all.json',
        'C2 CLS-Text': 'cls-text.json',
        'P1 Patch-All': 'patch-all-large.json',
        'P2 Patch-Text': 'patch-text-large.json',
    }
    resource = {name: load(EVIDENCE / 'resource' / file) for name, file in selected_files.items()}
    # Keep the empirically faster tested microbatch for each exact candidate.
    candidates = {}
    for name, raw in resource.items():
        peak_allocated = max(rank['peak_allocated_gib'] for rank in raw['ranks'])
        peak_reserved = max(rank['peak_reserved_gib'] for rank in raw['ranks'])
        pair_seconds = raw['projected_pair_compute_seconds_per_full_step_per_rank']
        p90_projected = (raw['pairs_per_full_step_per_rank'] /
                         (raw['args']['pair_batch'] / raw['p90_seconds']))
        candidates[name] = {
            'status': 'resource_infeasible',
            'visual_mode': raw['args']['visual_mode'],
            'readout_mode': raw['args']['readout_mode'],
            'visual_tokens': raw['visual_token_count'],
            'joint_sequence_length': raw['sequence_length'],
            'new_parameters': actual['projection_parameters'],
            'updates': 0,
            'recall': None,
            'J_long': None,
            'pair_microbatch': raw['args']['pair_batch'],
            'microbatch_median_seconds': raw['median_seconds'],
            'microbatch_p90_seconds': raw['p90_seconds'],
            'measured_pairs_per_second_per_rank': raw['measured_pairs_per_second_per_rank'],
            'projected_pair_only_step_median_seconds_lower_bound': pair_seconds,
            'projected_pair_only_step_p90_seconds_lower_bound': p90_projected,
            'projected_500_update_wall_hours_lower_bound': pair_seconds * 500 / 3600,
            'projected_500_update_gpu_hours_lower_bound': pair_seconds * 500 * 4 / 3600,
            'measured_pair_benchmark_peak_allocated_gib': peak_allocated,
            'measured_pair_benchmark_peak_reserved_gib': peak_reserved,
            'forward_flops_per_pair': raw['forward_flops_per_pair'],
            'forward_flops_per_full_step_per_rank': raw['forward_flops_per_full_step_per_rank'],
            'resource_gate_step_seconds': 30,
            'resource_gate_peak_allocated_gib': 65,
            'failure_reason': 'pair-only measured projection exceeds the complete-step time gate',
            'benchmark_git_head': raw['git_head'],
            'initial_sha256': raw['init_sha256'],
            'projection_sha256': raw['projection_sha256'],
            'rank_measurements': raw['ranks'],
        }
    refs = {}
    for name, group, new_parameters in (
            ('B0 T-fast', 'T-fast', 0), ('B1 Residual-CLS TI-fast', 'TI-fast', 114688)):
        metrics = reference['metrics'][group]
        refs[name] = {
            'status': 'completed_reference', 'updates': 500,
            'new_parameters': new_parameters, 'metrics': metrics,
            'J_long': j_long(metrics),
            'stable_step_median_seconds': reference['groups'][group]['timing'][
                'stable_step_max_rank_median_seconds'],
            'peak_allocated_gib': max(x['peak_allocated_gib'] for x in
                                      reference['groups'][group]['acceptance']['ranks']),
            'checkpoint_sha256': reference['groups'][group]['checkpoint_step500_sha256'],
            'bare_sha256': reference['groups'][group]['export_check']['bare_sha256'],
        }
    source_paths = [
        'model/model_longclip.py',
        'model/nested_joint_input.py',
        'train/train_nested_joint_input.py',
        'tests/test_nested_joint_input.py',
        'tests/joint_input_ddp_worker.py',
        'experiments/nest_clip_v1/mask_design_search_v1/benchmark_pairs.py',
        'experiments/nest_clip_v1/mask_design_search_v1/validate_actual.py',
        'experiments/nest_clip_v1/mask_design_search_v1/run.sh',
        'experiments/nest_clip_v1/mask_design_search_v1/summarize.py',
    ]
    source_hashes = {path: sha256(REPO / path) for path in source_paths}
    config_hashes = {
        path.name: sha256(path) for path in sorted((HERE / 'configs').glob('*.json'))
    }
    result = {
        'schema_version': 1,
        'date_utc': '2026-09-29',
        'branch': 'codex/nest-mask-design-search-v1',
        'base_commit': preflight['base_commit'],
        'resource_benchmark_commit': preflight['implementation_commit'],
        'initial_sha256': preflight['initial_sha256'],
        'source_sha256': source_hashes,
        'config_sha256': config_hashes,
        'evaluation_policy': preflight['evaluation_policy'],
        'excluded_evaluations': preflight['excluded'],
        'resource_gate': preflight['resource_gate'],
        'all_new_candidates_resource_infeasible': True,
        'formal_smoke_started': False,
        'formal_500_started': False,
        'new_candidate_evaluation_started': False,
        'references': refs,
        'candidates': candidates,
        'validation': {
            'unit_tests': {'passed': True, 'count': 23},
            'actual_vit': actual,
            'two_rank_nccl': ddp,
            'max_two_rank_gradient_error': max(
                case['max_gradient_error']['error'] for case in ddp['cases']),
            'max_adamw_update_error': max(
                case['max_adamw_update_error']['error'] for case in ddp['cases']),
            'adamw_note': ('large update differences are confined to explicitly checked '
                           'near-zero attention key-bias null directions'),
        },
        'recommendations': [
            {
                'candidate': 'B1 Residual-CLS TI-fast',
                'role': 'retain as the effect-and-cost balanced joint-mask candidate',
                'reason': ('completed 500-step evidence improves J_long over T-fast while retaining '
                           'about 2.0 s/step; every exact JointInput candidate fails the 30 s gate'),
            }
        ],
        'future_compressed_options_not_run': [
            'factorized or low-rank residual conditioning (the existing B1 design)',
            'frozen-size cross-attention with a small learned latent set',
            'predeclared text-token compression or patch-token selection in a new experiment',
        ],
    }
    (HERE / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')

    lines = [
        '# NEST mask design search v1: resource-gate report', '',
        'Date: 2026-09-29 UTC', '',
        'This round implemented the exact input-level JointInput definitions for CLS/Patch visual '
        'tokens and All/Text readout. No matching prior implementation or formal result was found. '
        'All four new candidates failed the frozen compute gate before smoke training, so no new '
        '500-step run or retrieval evaluation was started.', '',
        '## Decision', '',
        'All new candidates are `resource_infeasible`. The measured pair path alone projects to '
        'roughly 11.6-11.7 minutes per CLS step and 22.1-22.3 minutes per Patch step, before image/text '
        'encoding, feature gather, logging, checkpoint saving, or DataLoader work. The gate is 30 '
        'seconds for the complete step. Larger safe microbatches did not materially improve CLS '
        'throughput and only modestly improved Patch throughput.', '',
        'The full pair matrix contains 1,572,864 joint sequences per rank per regular step: three '
        'views, two retrieval directions, 256 local queries, and 1024 global candidates. Direct '
        'forward work is 2.664 PFLOP/rank for CLS and 5.029 PFLOP/rank for Patch. Even exact reuse of '
        'modality-independent projections cannot remove the pair-dependent MLP over interacted text '
        'tokens, so this is a compute limitation rather than a tensor-materialization mistake.', '',
        '## Resource matrix', '',
        '| Candidate | Tokens | Readout | New params | Microbatch | Pair throughput/rank | '
        'Projected pair-only step median/P90 | Measured peak alloc/reserved | 500-step wall lower bound | Status |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for name, row in candidates.items():
        lines.append(f"| {name} | {row['visual_tokens']} | {row['readout_mode']} | "
                     f"{row['new_parameters']:,} | {row['pair_microbatch']} | "
                     f"{row['measured_pairs_per_second_per_rank']:.1f}/s | "
                     f"{row['projected_pair_only_step_median_seconds_lower_bound']:.1f}/"
                     f"{row['projected_pair_only_step_p90_seconds_lower_bound']:.1f} s | "
                     f"{row['measured_pair_benchmark_peak_allocated_gib']:.2f}/"
                     f"{row['measured_pair_benchmark_peak_reserved_gib']:.2f} GiB | "
                     f"{row['projected_500_update_wall_hours_lower_bound']:.1f} h | resource_infeasible |")
    lines += ['',
        'The memory figures above are pair-microbenchmark peaks, not complete training-step peaks. '
        'They show that chunking controls resident tensors; they do not establish full-step memory. '
        'A complete step was not launched because the pair-only time lower bound already exceeds the '
        'gate by 23x for CLS and 44x for Patch.', '',
        '## Correctness and implementation checks', '',
        '- ViT-B/16 returns `X_I=[B,197,768]`, post-LN/pre-projection CLS `[B,768]`, and patches '
        '`[B,196,768]` from one visual forward. Native `z` and `h_cls @ visual.proj` match exactly.',
        '- Joint lengths are exactly 249 and 444; Text readout pools transformed text tokens after '
        'the shared MaskNetwork block. No residual JointMaskAdapter is called.',
        '- The shared projection A has 393,216 parameters and identical initialization SHA256 '
        f"`{actual['projection_state_sha256']}` for all candidates.",
        '- Changing the image while holding text fixed changes mask probabilities in all four modes. '
        'At initialization, All pooling assigns visual-token mass 0.0031 for CLS and 0.2688 for Patch.',
        '- Condition hidden states are detached before A; A and the original MaskNetwork receive '
        'finite gradients. Native CLIP state remains strictly loadable, and bare-student export omits A.',
        '- 23 focused/regression tests pass. The two-rank NCCL reference covers all-valid, a rank with '
        'zero valid samples, V=0, V=1, tail batches, checkpoint on/off, all four structures, named '
        'gradient None states, and AdamW one-step comparison.',
        f"- Maximum two-rank gradient error is {result['validation']['max_two_rank_gradient_error']:.7f}. "
        f"Maximum AdamW update error is {result['validation']['max_adamw_update_error']:.7f}, confined "
        'to verified near-zero attention key-bias null directions.', '',
        '## Existing 500-step references', '',
        '| Model | Method | New params | Updates | J_long | Stable step median | Peak allocated |',
        '|---|---|---:|---:|---:|---:|---:|',
        f"| B0 T-fast | text-only NEST | 0 | 500 | {100*refs['B0 T-fast']['J_long']:.2f} | "
        f"{refs['B0 T-fast']['stable_step_median_seconds']:.3f} s | {refs['B0 T-fast']['peak_allocated_gib']:.2f} GiB |",
        f"| B1 Residual-CLS TI-fast | low-rank residual CLS condition | 114,688 | 500 | "
        f"{100*refs['B1 Residual-CLS TI-fast']['J_long']:.2f} | "
        f"{refs['B1 Residual-CLS TI-fast']['stable_step_median_seconds']:.3f} s | "
        f"{refs['B1 Residual-CLS TI-fast']['peak_allocated_gib']:.2f} GiB |", '',
        '`J_long` is the frozen mean of Urban I2T/T2I R@1 and DOCCI I2T/T2I R@1. '
        'B1 improves it by +0.46 pp over B0 in the existing single-seed 500-step run. New candidates '
        'have no Recall or J_long entries because training and evaluation were correctly skipped.', '',
        '## Recommendation', '',
        'Retain **B1 Residual-CLS TI-fast** as the only next-stage candidate from this comparison. '
        'It has completed quality evidence and stays near 2 seconds/step. No exact JointInput candidate '
        'should advance under the current 4xA100 budget.', '',
        'If input-level interaction is revisited, define a new compressed experiment in advance: a '
        'small latent cross-attention set, explicit text-token compression, or patch-token selection. '
        'Those alter the current experiment and were not run here. No 3651-step training, extra seed, '
        'learning-rate scan, Long-DCI, or DCI Full evaluation was started.', '',
        '## Reproduction and artifacts', '',
        '- `run.sh tests` runs focused tests and the two-rank NCCL reference.',
        '- `run.sh validate` checks the real ViT/token definitions and conditioning diagnostics.',
        '- `run.sh resource C1|C2|P1|P2` repeats the four-rank resource benchmark.',
        '- `RESULTS.json` is the machine-readable status and evidence index, including source and '
        'configuration SHA256 values.',
        '- Raw small resource JSON and validation logs are under `evidence/`. No weights, data, or '
        'large profiler trace is committed.', '',
    ]
    (HERE / 'RESOURCE_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'results': str(HERE/'RESULTS.json'),
                      'report': str(HERE/'RESOURCE_REPORT.md')}, indent=2))


if __name__ == '__main__':
    main()
