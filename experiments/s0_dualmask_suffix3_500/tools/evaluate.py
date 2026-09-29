"""Run the original frozen six-protocol evaluators with explicit local data paths."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

from common import BUNDLE, sha, verify_repo

DOCS = BUNDLE.parents[1]
PROTOCOLS = {
    'flickr_test1k': ('flickr30k_test1k.jsonl', 'flickr30k/images', '113dbc616ca66db9400107ed4b97b56d33adae3c18608225902d97ee943600dc', 1000, 5000),
    'docci': ('docci_test.jsonl', 'docci/images', 'e852a96b4efb9fa6585fd70b2686cb4e456409b1144c4a0e3c7bd8a24a36cb11', 5000, 5000),
    'dci': ('dci_full.jsonl', 'dci/images', '14530fb8bf3c7b4a75bb451412d4562c548f9ca1bdbe1fdbf39d6ee30fb4de24', 7805, 7805),
    'long_dci': ('long_dci_reconstructed.jsonl', 'dci/images', '8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b', 7602, 7602),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--export-report', type=Path, required=True)
    parser.add_argument('--coco-root', type=Path, required=True)
    parser.add_argument('--urban-root', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    args = parser.parse_args()
    verify_repo(args.repo)
    for name in ['repo', 'checkpoint', 'export_report', 'coco_root', 'urban_root', 'benchmark_root', 'output_dir']:
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists():
        parser.error('Choose a new evaluation directory')
    identity = json.loads(args.export_report.read_text())
    weight_sha = sha(args.checkpoint)
    if identity.get('status') != 'PASS' or identity.get('completed_steps') != 500 or identity.get('bare_sha256') != weight_sha:
        raise RuntimeError('Bare-student identity does not match its strict export report')
    native_bundle = DOCS / 'experiments/s0_dualmask_masked_3epoch'
    native = native_bundle / 'tools/native_export_eval.py'
    extended = DOCS / 'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py'
    if sha(extended) != '5942fe84cb1befe68019d6344d7ae9fede8d453d34c9283cfafe1796ee660319':
        raise RuntimeError('Historical extended evaluator changed')
    frozen = json.loads((native_bundle / 'manifests/frozen_evaluators.json').read_text())
    for relative, value in frozen.items():
        if sha(native_bundle / 'reference_eval' / Path(relative).name) != value['sha256']:
            raise RuntimeError('Frozen evaluator changed: ' + relative)
    if sha(args.repo / 'tools/eval_urban1k_cls.py') != frozen['tools/eval_urban1k_cls.py']['sha256']:
        raise RuntimeError('Student loader changed')
    for manifest, _, expected, _, _ in PROTOCOLS.values():
        if sha(args.benchmark_root / 'manifests' / manifest) != expected:
            raise RuntimeError('Manifest differs from the published protocol: ' + manifest)
    annotation = args.coco_root / 'annotations/captions_val2017.json'
    if sha(annotation) != 'afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9':
        raise RuntimeError('COCO annotation identity mismatch')
    gpus = [x.strip() for x in args.gpus.split(',') if x.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        parser.error('Specify one or more distinct GPU IDs')
    available = queue.Queue()
    for gpu in gpus:
        available.put(gpu)
    args.output_dir.mkdir(parents=True)

    def evaluate(protocol):
        gpu = available.get()
        try:
            output = args.output_dir / protocol
            output.mkdir()
            result = output / (protocol + '.json')
            if protocol in ['coco', 'urban']:
                command = [sys.executable, str(native), '--repo', str(args.repo), '--action', protocol,
                           '--checkpoint', str(args.checkpoint), '--device', 'cuda:0', '--report', str(result)]
                command += ['--coco-root', str(args.coco_root)] if protocol == 'coco' else ['--urban-root', str(args.urban_root)]
            else:
                manifest, image_root, _, _, _ = PROTOCOLS[protocol]
                command = [sys.executable, str(extended), '--checkpoint', str(args.checkpoint),
                           '--device', 'cuda:0', '--batch-size', '64', '--output-dir', str(output),
                           f'{protocol}:{args.benchmark_root / "manifests" / manifest}:{args.benchmark_root / image_root}']
            (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(args.repo),
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
            started = time.monotonic()
            with (output / 'console.txt').open('w') as log:
                process = subprocess.run(command, cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            (output / 'exitcode.txt').write_text(str(process.returncode) + '\n')
            if process.returncode:
                raise RuntimeError(f'{protocol} failed; see {output}')
            data = json.loads(result.read_text())
            if data['checkpoint_sha256'] != weight_sha:
                raise RuntimeError('Evaluator used a different checkpoint')
            if protocol == 'urban' and data['dataset']['pair_caption_sha256'] != '3b0b2a3b743ed6011fccc72f8a9777bec35b44cadf2483f707fc5ef317da996c':
                raise RuntimeError('Urban pair/caption identity mismatch')
            if protocol in PROTOCOLS:
                _, _, expected, ni, nt = PROTOCOLS[protocol]
                if data['manifest_sha256'] != expected or (data['n_images'], data['n_captions']) != (ni, nt):
                    raise RuntimeError('Extended protocol identity mismatch')
            return {'protocol': protocol, 'physical_gpu': gpu, 'exit_code': 0, 'seconds': time.monotonic() - started, 'result': str(result)}
        finally:
            available.put(gpu)

    reports = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in concurrent.futures.as_completed([pool.submit(evaluate, x) for x in ['coco', 'urban', *PROTOCOLS]]):
            reports.append(future.result())
            (args.output_dir / 'progress.json').write_text(json.dumps(reports, indent=2) + '\n')
    (args.output_dir / 'summary.json').write_text(json.dumps({'status': 'COMPLETE', 'checkpoint_sha256': weight_sha,
        'evaluations': reports, 'extended_evaluator_sha256': sha(extended), 'frozen_evaluators': frozen}, indent=2) + '\n')


if __name__ == '__main__':
    main()
