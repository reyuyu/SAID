"""Frozen E2-Uniform continuation: 1217 -> 2434 -> 3651 -> 4868.

This controller only supplies isolated lineage/configuration checks around the
reviewed continuation protocol.  Production training, loss, sampling and
evaluation code are unchanged.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

from recovery import hns_s12_2434_validation as protocol
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, dump, now, sha

PROJECT = Path('/opt/data/private/lklk/SAID')
BRANCH = 'experiment/hns-s12-uniform-full4868-v1'
PARENT_BRANCH = 'origin/experiment/hns-s12-uniform-e1-1217-v1'
PARENT = PROJECT / 'runtime/SAID-nest-clip-v1/hns-s12-uniform-e1-1217-v1/step1217/training/step001217.pt'
PARENT_SHA = 'd276813e12b1b6da9a1f0247d9bbe13c6471276b45e852e06b149b9ad3cb5ead'
BASE_EXP = ROOT / 'experiments/nest_clip_v1/hns_s12_uniform_e1_1217_v1'
BASE_RUN = PROJECT / 'runtime/SAID-nest-clip-v1/hns-s12-uniform-e1-1217-v1'
EXP = ROOT / 'experiments/nest_clip_v1/hns_s12_uniform_full4868_v1'
RUN = PROJECT / 'runtime/SAID-nest-clip-v1/hns-s12-uniform-full4868-v1'
ENTRY = 'recovery.hns_s12_uniform_full4868'
TARGETS = (2434, 3651, 4868)
CODE = (ENTRY.replace('.', '/') + '.py', 'tests/test_hns_s12_uniform_full4868.py')
ORIGINAL_REPORT = protocol.report
ORIGINAL_CONFIGURE = protocol.configure
ORIGINAL_TRAINING_COMMAND = protocol.training_command


def segment(target):
    assert target in TARGETS
    return RUN / f'step{target}'


def phase(target):
    assert target in TARGETS
    return protocol.local.IMAGES.parent / f'formal-hns-s12-uniform-full4868-v1-{target}'


def predecessor(target):
    assert target in TARGETS
    if target == 2434:
        return PARENT, 1217
    previous = target - 1217
    return segment(previous) / 'training' / f'step{previous:06d}.pt', previous


def frozen(cfg):
    expected = protocol.common.read(BASE_EXP / 'config.json')
    assert all(cfg.get(k) == v for k, v in expected.items()), 'E2-Uniform config drift'
    assert tuple(cfg[k] for k in protocol.common.MACRO_KEYS) == (10.0, 1.2, 1.0)
    assert cfg['view_weights'] == [1.35, 1.35, 0.3]
    assert cfg['view_sparsity_weights'] == [5 / 3, 5 / 3, 5 / 3]
    assert cfg['hns_enabled'] and cfg['inclusion_max'] == 0
    assert cfg['sampling_mode'] == 'nested_detail_d3' and cfg.get('hns_beta', [2, 2]) == [2, 2]
    assert not cfg.get('hns_detach_child', False) and not cfg.get('hns_half_after500', False)


def load_git_json(ref, path):
    blob = subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=ROOT)
    return json.loads(blob), dict(branch=ref, commit=protocol.common.git('rev-parse', ref),
                                path=path, sha256=hashlib.sha256(blob).hexdigest())


def configure(target=None):
    if target is not None:
        protocol.local.RUN = segment(target)
        protocol.local.PHASE = phase(target)
        protocol.local.CONFIG = EXP / 'config.json'


def training_command(target):
    parent, _ = predecessor(target)
    return protocol.common.torchrun(ENTRY, '--worker', '--target', target,
        '--config', EXP / 'config.json', '--init-state', STEP0, '--resume', parent,
        '--index-dir', protocol.local.INDEX, '--image-root', protocol.local.IMAGES,
        '--output-dir', segment(target) / 'training', '--run-type', 'formal',
        '--max-updates', target)


def prepare():
    from tools.eval_five_parallel import require_gpu_idle
    require_gpu_idle({0, 1, 2, 3})
    assert protocol.common.git('branch', '--show-current') == BRANCH
    assert not EXP.exists() and not RUN.exists(), 'Refuse overwrite or implicit retry'
    assert protocol.common.git('rev-parse', PARENT_BRANCH) == 'f7b1c94299ca464569b40e4bdde2eec615317253'
    assert sha(STEP0) == STEP0_SHA
    proof = protocol.identity(PARENT, 1217)
    assert proof['sha256'] == PARENT_SHA
    parent_result = protocol.common.read(BASE_EXP / 'step1217/RESULTS.json')
    assert parent_result['completed_steps'] == 1217 and parent_result['checkpoint']['sha256'] == PARENT_SHA
    assert parent_result['strict_export']['passed'] and parent_result['evaluation_checkpoint_immutable']
    for target in TARGETS:
        prior, start = predecessor(target)
        assert start in (1217, 2434, 3651)
        (EXP / f'step{target}').mkdir(parents=True)
        segment(target).mkdir(parents=True)
        dump(segment(target) / 'resume-reference.json', protocol.resume_reference(start))
    ready = protocol.common.read(protocol.local.IMAGES.parent / 'full-ready.json')
    assert ready['status'] == 'LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    evaluator = protocol.common.evaluator_proof()
    e2cfg = protocol.common.read(BASE_EXP / 'config.json')
    frozen(e2cfg)
    own_curve = {'500': protocol.common.read(BASE_EXP / 'REFERENCES.json')['models']['E2_Uniform_500']['1217'],
                 '1217': parent_result}
    models = {}
    refs = []
    s12, receipt = load_git_json('origin/experiment/hns-s12-2434-validation-v1',
        'experiments/nest_clip_v1/hns_s12_2434_validation_v1/step2434/RESULTS.json')
    models['HNS_S12'] = {'2434': s12}
    refs.append(receipt)
    full, receipt = load_git_json('origin/experiment/hns-s12-full4868-v1',
        'experiments/nest_clip_v1/hns_s12_full4868_v1/REFERENCES.json')
    refs.append(receipt)
    for name in ('HNS_v1', 'D3_Balanced'):
        models[name] = {k: full['models'][name][k] for k in ('3651', '4868')}
    full_results, receipt = load_git_json('origin/experiment/hns-s12-full4868-v1',
        'experiments/nest_clip_v1/hns_s12_full4868_v1/RESULTS.json')
    refs.append(receipt)
    for step in ('3651', '4868'):
        models['HNS_S12'][step] = full_results['nodes'][step]
    # HNS and Balanced same-node references at 2434 are already in the parent report.
    prior2434, receipt = load_git_json('origin/experiment/hns-s12-2434-validation-v1',
        'experiments/nest_clip_v1/hns_s12_2434_validation_v1/REFERENCES.json')
    refs.append(receipt)
    for name in ('HNS_v1', 'D3_Balanced'):
        models.setdefault(name, {})['2434'] = prior2434['models'][name]['2434']
    EXP.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    dump(EXP / 'config.json', e2cfg)
    dump(EXP / 'RESUME_PROVENANCE.json', dict(**proof, parent_branch=PARENT_BRANCH,
        parent_commit=protocol.common.git('rev-parse', PARENT_BRANCH), parent_sha256=PARENT_SHA,
        production_source_changes=[], local_only=True, NFS_fallback=False,
        checkpoint_lineage='E2-Uniform@500 -> E2-Uniform@1217 -> this trajectory'))
    dump(EXP / 'REFERENCES.json', dict(models=models, own_curve=own_curve, provenance=refs,
        matched_nodes_only=True, E2_Uniform_1217_primary=parent_result))
    paths = protocol.local.path_proof()
    dump(RUN / 'local-path-proof-5000.json', paths)
    dump(EXP / 'LOCAL_ONLY_PROOF.json', dict(passed=paths['passed'], count=paths['count'],
        image_root=str(protocol.local.IMAGES), NFS_fallback=False,
        raw_path=str(RUN / 'local-path-proof-5000.json')))
    dump(EXP / 'PLAN.json', dict(order=['resume1217->2434','evaluate/report2434',
        'resume2434->3651','evaluate/report3651','resume3651->4868','evaluate/report4868','STOP'],
        targets=list(TARGETS), initial_updates=1217, horizon=4868, macro=[10,1.2,1],
        view_weights=[1.35,1.35,.3], raw_sparsity_ratio=[1,1,1],
        normalized_sparsity_weights=[5/3]*3, effective_sparsity_coefficients=[2,2,2],
        beta=[2,2], Hard_ST=True, no_SG=True, soft_inclusion=0, K=3,
        ramp200_unchanged=True, local_only=True, full_restore=True,
        evaluation_mapping=dict(coco=0, docci=1, long_dci=2, flickr=3, urban=3),
        automatic_continuation=False, other_arms=False))
    tests = [str(PROJECT/'.venv/bin/python'), '-m', 'pytest', '-q',
             'tests/test_hns_s12_uniform_full4868.py']
    log = RUN / 'cpu-tests.log'
    with log.open('w') as handle:
        check = subprocess.run(tests, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    dump(EXP / 'CPU_TESTS.json', dict(passed=check.returncode == 0, returncode=check.returncode,
        command=tests, summary=log.read_text(), checked_utc=now()))
    assert check.returncode == 0
    protocol.state('PREPARED')


def report(target, supervisor, result):
    from recovery import nested_d3_local_search as search
    old = search.ARMS.get('S12')
    search.ARMS['S12'] = dict(mode='nested_detail_d3', weights=[5/3, 5/3, 5/3])
    try:
        ORIGINAL_REPORT(target, supervisor, result)
    finally:
        if old is not None:
            search.ARMS['S12'] = old


def combined(completed):
    reference_doc = protocol.common.read(EXP/'REFERENCES.json')
    refs = reference_doc['models']
    existing = []
    if (EXP/'step2434/RESULTS.json').exists():
        existing = [2434]
    results = {str(n): protocol.common.read(EXP/f'step{n}/RESULTS.json') for n in sorted(set(existing + list(completed)))}
    own = reference_doc['own_curve']
    curve = {'500': own['500'], '1217': own['1217']}
    curve.update({str(n): results[str(n)] for n in sorted(set(existing + list(completed)))})
    final = 4868 in completed
    dump(EXP/'RESULTS.json', dict(status='COMPLETED' if final else 'PARTIAL', nodes=results,
        learning_curve=curve, references=refs, frozen_macro=[10,1.2,1],
        sparsity_ratio=[1,1,1], stop_updates=4868, automatic_continuation=False))
    lines = ['# E2-Uniform full 4868 validation', '',
        'Uniform sparsity [5/3,5/3,5/3], effective [2,2,2], fixed macro loss [10,1.2,1].', '',
        '| Step | Model | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |',
        '|---:|---|---:|---:|---:|---:|---|---:|']
    for n in ('500','1217') + tuple(str(x) for x in sorted(set(existing + list(completed)))):
        item = curve[n]; q = protocol.common.quality(item)
        lines.append(f'| {n} | E2-Uniform | ' + ' | '.join(f'{q[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4')) +
                     f' | {q["Urban_I2T"]:.3f} / {q["Urban_T2I"]:.3f} | {(q["Urban_I2T"]+q["Urban_T2I"])/2:.3f} |')
        if n in results:
            for name in ('HNS_S12','HNS_v1','D3_Balanced'):
                if n in refs.get(name, {}):
                    b = protocol.common.quality(refs[name][n])
                    lines.append(f'| {n} | {name} | ' + ' | '.join(f'{b[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4')) +
                                 f' | {b["Urban_I2T"]:.3f} / {b["Urban_T2I"]:.3f} | {(b["Urban_I2T"]+b["Urban_T2I"])/2:.3f} |')
    for n in sorted(set(existing + list(completed))):
        node = results[str(n)]
        lines += ['', f'## Step {n}', '', '| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |', '|---|---|---|']
        for ds, dirs in node['metrics'].items():
            lines.append(f'| {ds} | ' + ' / '.join(f'{100*dirs["I2T"][k]:.6f}' for k in ('R@1','R@5','R@10')) +
                         ' | ' + ' / '.join(f'{100*dirs["T2I"][k]:.6f}' for k in ('R@1','R@5','R@10')) + ' |')
        for name, comp in node['comparisons'].items():
            lines += ['', f'{name} deltas (pp): {json.dumps(comp["quality_delta_pp"], sort_keys=True)}']
    lines += ['', 'All production training/loss/sampling/evaluator sources remain checkpoint-manifest identical.',
              'Every node uses complete optimizer/scheduler/RNG/sampler restoration and stops only at 4868.',
              'No fifth epoch, parameter search, arm combination or additional continuation is authorized.']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')


def configure_protocol():
    protocol.BRANCH = BRANCH; protocol.MOTHER = PARENT_BRANCH; protocol.PARENT = PARENT
    protocol.PARENT_SHA = PARENT_SHA; protocol.EXP = EXP; protocol.RUN = RUN
    protocol.ENTRY = ENTRY; protocol.TARGETS = TARGETS; protocol.CODE = CODE
    protocol.segment = segment; protocol.phase = phase; protocol.predecessor = predecessor
    protocol.frozen = frozen; protocol.configure = configure; protocol.training_command = training_command
    protocol.prepare = prepare; protocol.report = report; protocol.combined = combined


def continue_existing():
    """Continue from an already completed and evaluated2434 node.

    The first node is never retrained; this mode is used only after the
    report-only reference-table repair for the existing2434 checkpoint.
    """
    assert protocol.common.read(EXP/'STATE.json')['status'] == 'PREPARED'
    protocol.TARGETS = (3651, 4868)
    protocol.run()


def main():
    configure_protocol()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue-existing', action='store_true')
    args, _ = parser.parse_known_args()
    if args.continue_existing:
        continue_existing()
    else:
        protocol.main()


if __name__ == '__main__':
    main()
