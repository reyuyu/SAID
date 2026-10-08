"""Prove shared controllers change scheduling only, against the original source."""
import ast
import hashlib
import json
import subprocess

from tools import eval_five_parallel as ev

BASELINE='f7062162a065c5cb6b141a2edfed969d3692f2da'
CONTROLLERS=(
    ('experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_full.py','Supervisor'),
    ('experiments/nest_clip_v1/balanced_hparam_search_v1/search.py','Search'),
)


def method(tree,cls):
    return next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls
                for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')


def is_scheduler(node):
    return isinstance(node,ast.Expr) and any(
        isinstance(n,ast.Constant) and n.value=='tools.eval_five_parallel' for n in ast.walk(node))


def remove_scheduling(body):
    result=[]
    for node in body:
        # The original two evaluator launch loops and their dataset-root alias.
        if isinstance(node,ast.For) and any(isinstance(n,ast.Constant) and n.value in
            ('tools.eval_nest_native',ev.EXTENDED) for n in ast.walk(node)):continue
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bench' for t in node.targets):continue
        if is_scheduler(node):continue
        if isinstance(node,ast.If):node.body=remove_scheduling(node.body)
        result.append(node)
    return result


def test_shared_integration_preserves_export_aggregation_and_all_other_code():
    for path,cls in CONTROLLERS:
        old=ast.parse(subprocess.check_output(['git','show',BASELINE+':'+path],cwd=ev.ROOT,text=True))
        new=ast.parse((ev.ROOT/path).read_text())
        om,nm=method(old,cls),method(new,cls)
        scheduler=[n for n in ast.walk(nm) if is_scheduler(n)]
        assert len(scheduler)==1
        strings=[n.value for n in ast.walk(scheduler[0]) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        assert all(flag in strings for flag in ('--checkpoint','--training-checkpoint','--output-dir','--assets-root'))
        # All export/verify, checkpoint integrity, existing aggregate arithmetic,
        # training and controller code must be structurally identical.
        om.body=remove_scheduling(om.body);nm.body=remove_scheduling(nm.body)
        assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)


def test_validation_passed_before_default_integration_and_evaluators_unchanged():
    root=ev.ROOT/'engineering/parallel_five_eval_v1'
    comparison=json.loads((root/'COMPARISON.json').read_text())
    integration=json.loads((root/'INTEGRATION.json').read_text())
    assert comparison['passed'] and comparison['metrics_exact'] and comparison['scores_exact']
    assert integration['validation_passed_before_integration']
    assert integration['validation_completed_utc']<=integration['integrated_utc']
    for path,digest in comparison['unchanged_evaluator_source_sha256'].items():
        assert hashlib.sha256((ev.ROOT/path).read_bytes()).hexdigest()==digest
        original=subprocess.check_output(['git','show',BASELINE+':'+path],cwd=ev.ROOT)
        assert hashlib.sha256(original).hexdigest()==digest
