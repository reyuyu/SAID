"""Collect completed Phase A JSON receipts without running any GPU work."""
import argparse
import json
import re
from pathlib import Path

from recovery.e2_uniform_posthoc_audit import write_new
from recovery.visual_patch_gradient_phase_a import NODES, ROOT, sha


def stopped_preflight(runtime,destination):
    """Publish only what the stopped execution proves; do not run more probes."""
    destination.mkdir(parents=True,exist_ok=False)
    text=(runtime/'node500-validated.log').read_text()
    failed=re.findall(r'\[rank(\d)\]: AssertionError: .*relative_L2_error.*?(0\.\d+)',text)
    assert {int(rank) for rank,_ in failed}=={0,1,2,3}
    assert all(float(error)>.01 for _,error in failed)
    error=float(failed[0][1]);assert len({value for _,value in failed})==1
    identities={str(node):dict(path=str(path),sha256=sha(path),expected_sha256=expected)
                for node,(path,expected) in NODES.items()}
    assert all(x['sha256']==x['expected_sha256'] for x in identities.values())
    from train.train_nested_semantic_mask import code_manifest
    import torch
    for node,(path,_) in NODES.items():
        payload=torch.load(path,map_location='cpu',weights_only=False)
        assert payload['config']['code_sha256']==code_manifest()
        del payload
    evidence=dict(conclusion='INCONCLUSIVE',status='STOPPED_AT_PREFLIGHT',checkpoint_identities=identities,
        source_manifest=code_manifest(),no_training=True,no_optimizer_created=True,no_optimizer_step=True,
        no_checkpoint_written=True,parameters_not_updated=True,global1024_status='NOT_RUN',node1217_status='NOT_RUN',
        log_sha256=sha(runtime/'node500-validated.log'),log_server_path=str(runtime/'node500-validated.log'),
        forward_evidence='All four ranks reached the additivity gate after exact-forward and state-invariance assertions; per-tensor snapshot JSON was not persisted by the initial probe before failure.',
        state_digest_values='UNVERIFIED: initial probe verified state equality in memory but did not persist hashes before the failed gate.',
        numerical_failure=dict(group='native_visual_backbone',relative_L2_error=error,tolerance=.01,passed=False,
            variant='UNVERIFIED: initial traceback did not label A versus B; do not infer.',ranks=[0,1,2,3]),
        checkpoint_sha256_before_and_after_match=True,
        formal_gradients='UNVERIFIED',increased_patch_path_utility='UNVERIFIED',retrieval_benefit='UNVERIFIED')
    write_new(destination/'VISUAL_GRAD_FORWARD_EQUIVALENCE.json',dict(**evidence,preflight_node=500,
        preflight_global_batch=16,forward_status='PASS_BY_EXECUTED_ASSERTIONS',
        forward_absolute_difference=0.,forward_relative_difference=0.,
        formal_forward_status='UNVERIFIED',detail='Strict exact checks ran for every captured forward tensor on four ranks before the failed gate. Full detailed tensors/shape metadata were not persisted.'))
    write_new(destination/'VISUAL_GRAD_COMPONENT_AUDIT.json',dict(**evidence,
        component_gradient_norms='UNVERIFIED: not persisted before stop',
        cosine_metrics='UNVERIFIED: not persisted before stop',
        finite_gradients_status='PASS_BY_EXECUTED_ASSERTIONS for four weighted components, both variants, four ranks on global16 only',
        original_native_regularizer_zero_gradient='UNVERIFIED: gate failed before explicit assertion',
        patches_A_B_route='UNVERIFIED: computed in memory but receipt not persisted'))
    write_new(destination/'VISUAL_GRAD_LAYERWISE_AUDIT.json',dict(**evidence,
        layerwise_metrics='UNVERIFIED: stopped during group acceptance before layerwise reporting'))
    write_new(destination/'VISUAL_GRAD_RESOURCE_AUDIT.json',dict(**evidence,
        per_variant_peak_memory_and_seconds='UNVERIFIED: computed in memory but receipt not persisted',
        OOM_observed=False,nonfinite_observed=False,NCCL_communication_failure_observed=False,
        process_exit=1,resource_note='torchrun propagated the assertion; NCCL cleanup warning is secondary to failed acceptance, not evidence of a transport fault',
        failures_preserved_locally=[str(runtime/x) for x in ('node500.log','node500-probe.log','node500-validated.log')]))
    write_new(destination/'STOPPED_PREFLIGHT_EVIDENCE.json',evidence)
    relevant=[line for line in text.splitlines() if 'AssertionError:' in line or 'gradient_component' in line]
    with (destination/'PREFLIGHT_LOG_EXCERPT.txt').open('x') as handle:handle.write('\n'.join(relevant)+'\n')
    lines=['# Visual Patch conditional gradient: Phase A stopped preflight','',
        'Conclusion: `INCONCLUSIVE`. The gradient acceptance failed before any formal global1024 run. No Phase B training is authorized or started.','',
        '## What ran','',
        'CPU tests and checkpoint identities passed. GPU0–3 ran a global16 (4/rank) preflight at fixed E2-Uniform@500, with native BF16 encoder autocast/checkpointing and original FP32 masks/fusion. Both variants used identical model parameter objects, frozen values, input tensors and RNG. Only the visual-adapter input detach was removed in the local B override; production code and text detach remained unchanged. No optimizer was constructed; no parameters were updated and no checkpoint was saved.','',
        'The full256/rank next501/502 inputs were prepared; all original JSONL sample IDs, F/Dall/D3 text/token summaries, K and selected-index fields matched before slicing to preflight4/rank. The initial comparison failed because the probe omitted the observer detail-index field and compared Python histogram integer keys to JSONL string keys; read-only CPU reconstruction established exact original serialized-field agreement. A regression test was added. The subsequent GPU preflight failed the independent gradient acceptance below. Original failure logs are preserved locally.','',
        '## Failed numerical acceptance','',
        f'All four ranks reported native visual backbone component-gradient additivity relative L2 error **{error:.9%}**, exceeding the predeclared **1%** BF16 tolerance. The probe uses true per-parameter component and total autograd gradients, native differentiable all-gather and mean four-rank reduction. FP32 downstream tolerance was predeclared0.03%. Tolerances were not widened after observing failure.','',
        'Separate BF16 backward passes can round shared encoder-path accumulations differently, so this discrepancy could be numerical. That explanation has not been independently established. It must not be labelled either a beneficial Patch signal or a harmful training mechanism. The failed traceback omitted the A/B variant, and numeric gradient vectors were not persisted before failure; this limitation is explicit in the JSON files.','',
        'The initial script checked all captured A/B forward tensors for bit-exact equality and checked complete state-digest equality and unchanged parameter structure after each backward pass, before reaching the failed gate. All four ranks reached that gate. Thus global16 forward equivalence and in-memory immutability checks passed, but detailed per-tensor receipts and literal state hashes were not persisted. No formal-batch conclusion is claimed. All gradient finite checks preceding the failed gate passed.','',
        'The probe has now been improved to persist numerical failure receipts before raising and cleanly release its process group. This reporting-only revision was CPU tested and was **not rerun on GPU**. It changes no production code, loss, precision or numerical threshold.','',
        '## Required research questions','',
        '| Question | Evidence and status |','|---|---|',
        '| Q1: What does detach block? | Production graph and CPU tests show that adapter-input Patch gradients are blocked, while the native CLS alignment path stays live. Real E2 regularizer-zero/path-norm receipts are UNVERIFIED because the preflight stopped before publishing them. |',
        '| Q2: How large, from which component? | UNVERIFIED; A/B components were differentiated, but norms were not persisted before failure. |',
        '| Q3: Cooperative, orthogonal or conflicting? | UNVERIFIED; no numerical cosines are reported. |',
        '| Q4: Similar across500/1217? | UNVERIFIED;1217 was not run after500 preflight failure. |',
        '| Q5: Extra memory/global1024 feasibility? | UNVERIFIED; no OOM in global16, but this does not establish global1024 feasibility. Per-variant memory/time receipts were not persisted. |',
        '| Q6: Proceed to500-step training? | No recommendation to train yet. First resolve component-gradient consistency in a separately authorized read-only probe without changing production precision/model/loss. |','',
        'A future user-approved Phase B, if numerical evidence supports it, would change only this single visual input detach on the frozen E2 protocol. This task does not start it. No retrieval evaluation or Urban feedback was used. Gradient direction alone cannot prove retrieval benefit.','',
        '## Immutable identities and stopping','']
    for node,item in identities.items():lines.append(f'- E2@{node}: `{item["sha256"]}`; before/after expected hashes match.')
    lines+=['','Both checkpoint manifests match the untouched production sources in the analysis worktree. Existing E2 reports, evaluator code and weights were not written. All formal gradient, layerwise, stability and resource claims remain UNVERIFIED. Scientific status is INCONCLUSIVE, not MECHANISM_UNFAVORABLE.','',
        'Reproduction scripts: `recovery/visual_patch_gradient_phase_a.py` (GPU; further execution requires user direction after this failed acceptance) and `recovery/collect_visual_patch_gradient_phase_a.py --stopped-preflight` (CPU-only report from saved failure logs).']
    with (destination/'VISUAL_GRAD_PHASE_A_REPORT.md').open('x') as handle:handle.write('\n'.join(lines)+'\n')


def collect(runtime, destination, conclusion):
    destination.mkdir(parents=True,exist_ok=False)
    records={};identities={}
    for node in (500,1217):
        folder=runtime/('node'+str(node))
        assert json.loads((folder/'COMPLETED.json').read_text())['passed']
        identities[str(node)]=dict(checkpoint=str(NODES[node][0]),sha256=sha(NODES[node][0]),
                                  expected_sha256=NODES[node][1],unchanged=True)
        assert identities[str(node)]['sha256']==NODES[node][1]
        for label in ('preflight16','global1024_next1','global1024_next2'):
            record=json.loads((folder/(label+'.json')).read_text());assert record['passed']
            records[str(node)+'_'+label]=record
    common=dict(passed=True,no_optimizer_created=True,no_optimizer_updates=True,no_new_checkpoint=True,
                checkpoints=identities,source_manifest=next(iter(records.values()))['source_manifest'])
    write_new(destination/'VISUAL_GRAD_FORWARD_EQUIVALENCE.json',dict(**common,records={
        k:dict(node=v['node'],global_batch=v['global_batch'],rank_comparisons=v['forward'],
               input_audit=v['input_audit'],same_RNG=v['same_RNG'],same_parameter_objects=v['same_parameter_objects'])
        for k,v in records.items()}))
    write_new(destination/'VISUAL_GRAD_COMPONENT_AUDIT.json',dict(**common,records={
        k:dict(node=v['node'],global_batch=v['global_batch'],groups=v['group_gradients'],patch_route=v['patch_route'])
        for k,v in records.items()}))
    write_new(destination/'VISUAL_GRAD_LAYERWISE_AUDIT.json',dict(**common,records={
        k:v['layerwise_gradients'] for k,v in records.items()}))
    write_new(destination/'VISUAL_GRAD_RESOURCE_AUDIT.json',dict(**common,records={
        k:dict(resources=v['resources'],gpu_identity=v['gpu_identity']) for k,v in records.items()},
        wall_time_definition='Per-variant forward, component autograd, mean gradient reductions, CPU copies and state digest; not training throughput',
        precision='Original encoder BF16 autocast; FP32 mask/fusion; TF32 disabled',
        first_launcher_error='torchrun --node abbreviation ambiguity; no worker/GPU computation started. Fixed child option name to --checkpoint-node. Original launcher stderr preserved locally.',
        initial_sampling_comparison_error='Stopped before forward at comparison of Python dict to historical JSONL. Probe initially omitted observer detail-index digest and integer histogram-key JSON serialization. CPU read-only reconstruction proved identical original serialized fields. Regression test added; fresh probe output preserves prior failure evidence. No actual sampling drift or NCCL communication failure.'))
    for node in (500,1217):
        write_new(destination/('NODE'+str(node)+'_PLAN.json'),json.loads((runtime/('node'+str(node))/'PLAN.json').read_text()))
    lines=['# Visual Patch condition gradients: Phase A','',
        'Conclusion: `'+conclusion+'`. This is a read-only mechanism study, not a retrieval result. No optimizer was created; no parameters or checkpoint files were updated. Production sources are byte-identical to the verified E2 checkpoints.','',
        '## Fixed protocol','',
        'E2-Uniform@500 and@1217; native BF16 encoder checkpointing, original FP32 fusion/masks, HNS no-SG beta2/2, fixed K3 and coefficients. A uses the exact production function; B replaces only `hidden.detach().float()` with `hidden.float()` on the same model instance. Text detach is retained. Parameter objects, complete state hashes, inputs and CPU/CUDA/Python RNG match. Four-rank differentiable gathering and mean all-reduced true parameter gradients follow the existing read-only audit.','',
        'Each node passed global16 (4/rank) preflight and two predeclared global1024 (256/rank) batches: next501/502 or1218/1219. Sample IDs, F/Dall/D3 text/token summaries, K and indices exactly match original E2 continuation logs. Native image augmentation is regenerated using the checkpoint worker generator on an isolated suffix; historical random pixel equality is not claimed. A/B see identical pixels. No model updates occur between batches.','',
        'All captured forward values are bit-exact on all four ranks: native image/text embeddings, visual adapter output and conditions, three probabilities and Hard-ST masks, three raw CEs, sparsity, hierarchy and weighted total loss.','',
        '## Native visual backbone gradients','',
        '| Node | Fixed batch | Base norm | New norm | Delta norm | Delta/base | cos(delta,base) | Added align norm | Added sparse norm | Added HNS norm |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,v in records.items():
        g=v['group_gradients']['native_visual_backbone']
        lines.append(f'| {v["node"]} | {v["batch"]} | '+ ' | '.join(f'{g[x]:.7g}' for x in
            ('g_base_norm','g_new_norm','delta_g_norm','delta_g_over_g_base','cosine_delta_base'))+' | '+
            ' | '.join(f'{g["added_component_norms"][x]:.7g}' for x in ('alignment','sparsity','hierarchy'))+' |')
    lines+=['','| Node/batch | cos(added align,base align) | cos(added sparse,base align) | cos(added HNS,base align) |',
            '|---|---:|---:|---:|']
    for key,v in records.items():
        g=v['group_gradients']['native_visual_backbone']
        lines.append('| '+key+' | '+' | '.join(f'{g[x]:.7g}' if g[x] is not None else 'undefined (zero vector)' for x in
            ('cosine_added_alignment_base_alignment','cosine_added_sparsity_base_alignment','cosine_added_hierarchy_base_alignment'))+' |')
    lines+=['','## Path and invariance evidence','',
        'Original regularizers have exactly zero native visual backbone gradients, while the existing CLS alignment gradient remains nonzero. The derivative to the exact Patch tensor entering the adapter is disconnected for all A losses and live for B; its per-rank norms are in COMPONENT_AUDIT. B therefore adds a Patch route without changing the native CLS forward or text gradient routing.','',
        'All weighted component gradients are finite. Their sum is checked against separately differentiated total gradients. BF16 native encoder additivity has a predeclared 1% relative L2 tolerance because separate backward passes round shared encoder-path accumulation; FP32 downstream retains the original 0.03% tolerance. This tolerance does not permit forward differences. Observed errors are reported below.','',
        'All non-native-visual parameter groups have exactly zero A/B gradient differences, component by component: native text backbone, text mask/shared pool, visual mask, visual adapter and fusion gate. Detailed Transformer layers and native visual projection/embedding/norm groups are in LAYERWISE_AUDIT.','',
        '| Node/batch | A visual additivity relative L2 | B visual additivity relative L2 |',
        '|---|---:|---:|']
    for key,v in records.items():
        g=v['group_gradients']['native_visual_backbone']['gradient_additivity']
        lines.append(f'| {key} | {g["A"]["relative_L2_error"]:.7g} | {g["B"]["relative_L2_error"]:.7g} |')
    lines+=['','## Resources','',
        '| Node/batch | A max allocated GiB | B max allocated GiB | Increase GiB | A wall seconds | B wall seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for key,v in records.items():
        r=v['resources'];a=max(x['peak_allocated_bytes'] for x in r['A'])/2**30;b=max(x['peak_allocated_bytes'] for x in r['B'])/2**30
        lines.append(f'| {key} | {a:.3f} | {b:.3f} | {b-a:+.3f} | {max(x["seconds"] for x in r["A"]):.3f} | {max(x["seconds"] for x in r["B"]):.3f} |')
    lines+=['','Wall durations include audit overhead and repeated component differentiation, not a training speed comparison. Peak reserved memory and every rank are in RESOURCE_AUDIT.','',
        '## Interpretation and limits','',
        'Gradient norms and cosines describe local derivatives on frozen checkpoints; they do not establish useful learning, Urban gains, generalization or an optimum regularization strategy. Four fixed formal batches from two early stages cannot establish training-wide stability, and small preflight batches are not formal global1024 evidence. The added alignment gradient is local-conditioned pair alignment, not an independent supervision source. Added sparsity/hierarchy gradients can compete with both the base alignment and the added alignment path.','',
        'Any Phase B requires user approval and must keep the same frozen E2 configuration, initialization, data flow and evaluation protocol, changing only this one visual input detach. Phase A does not authorize it. No Urban test feedback was used to select a gradient route.','',
        'Pinned checkpoint SHAs:']
    for node,info in identities.items():lines+=['',f'- E2@{node}: `{info["sha256"]}`.']
    with (destination/'VISUAL_GRAD_PHASE_A_REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--conclusion',choices=['MECHANISM_SUPPORTED','MECHANISM_UNFAVORABLE','INCONCLUSIVE'])
    p.add_argument('--stopped-preflight',action='store_true')
    args=p.parse_args()
    if args.stopped_preflight:stopped_preflight(args.runtime,args.output_dir)
    else:
        assert args.conclusion is not None
        collect(args.runtime,args.output_dir,args.conclusion)
