"""Build the six-protocol, budget-labelled comparison after all evaluations finish."""
import json
from pathlib import Path
import sys

REPO=Path('/root/lk_projects/SAID')
ROOT=Path('/root/lk_projects/SAID-nest-clip-v1')
DEST=ROOT/'randomk3epoch'
EV=DEST/'evidence'
sys.path.insert(0,str(REPO))
from train.nested_semantic_data import file_sha

def read(p):return json.loads(Path(p).read_text())
protocols=['coco','urban','flickr_test1k','docci','dci','long_dci']
audit=read(EV/'training-audit.json');assert audit['passed']
export=read(DEST/'A3-RandomK/export-check.json')
assert export['passed'] and export['optimizer_steps']==[3651]
assert export['image_max_abs']==export['text_max_abs']==0
old_export=read(ROOT/'randomk500/A3-RandomK/export-check.json')
raw={};executions={}
for step in (500,3651):
    raw[str(step)]={}
    digest=old_export['bare_sha256'] if step==500 else export['bare_sha256']
    for protocol in protocols:
        if step==500 and protocol in ('coco','urban'):
            result=read(ROOT/f'randomk500/A3-RandomK/{protocol}_native.json')
        else:
            folder=DEST/f'evaluation/step{step}'
            path=folder/f'{protocol}_native.json' if protocol in ('coco','urban') else folder/protocol/f'{protocol}.json'
            result=read(path)
            name=f'eval{step}-{protocol}';execution=read(EV/f'{name}.execution.json')
            assert execution['exit_code']==0
            executions[name]=execution
        assert result['checkpoint_sha256']==digest
        raw[str(step)][protocol]=result
baseline={name:{protocol:read(EV/f'baselines/{name}/{protocol}.json') for protocol in protocols}
          for name in ('clean3651','full3651')}
def recall(result,protocol,direction,k):
    key='image2text' if direction=='I2T' else 'text2image'
    if protocol=='coco':return result.get('metrics',result)[f'{key}_R{k}']
    if protocol=='urban':return result.get('urban1k',result)[key][f'R{k}']
    return result['metrics'][f'R@{k}'][direction]
comparison=[]
for protocol in protocols:
    for direction in ('I2T','T2I'):
        for k in (1,5,10):
            values={f'RandomK{step}':recall(raw[str(step)][protocol],protocol,direction,k) for step in (500,3651)}
            values.update({name:recall(results[protocol],protocol,direction,k) for name,results in baseline.items()})
            assert all(0<=x<=1 for x in values.values())
            comparison.append(dict(protocol=protocol,direction=direction,metric=f'R@{k}',raw=values,
                percent={key:100*v for key,v in values.items()},
                delta_pp={key:100*(values['RandomK3651']-values[key]) for key in ('RandomK500','clean3651','full3651')}))
for name in ('train3651','audit3651','export3651','verify-export3651'):
    execution=read(EV/f'{name}.execution.json');assert execution['exit_code']==0;executions[name]=execution
pre=read(EV/'preflight.json')
for spec in pre['protocols']:
    for step in ('500','3651'):
        v=raw[step][spec['name']]
        assert v['manifest_sha256']==spec['manifest_sha256']
        assert (v['n_images'],v['n_captions'])==(spec['n_images'],spec['n_captions'])
result=dict(code_commit=audit['code_commit'],updates=3651,epochs=3,horizon=3651,
    continuation_start=500,new_updates=3151,preflight=pre,audit=audit,export=export,executions=executions,
    evaluations=raw,baselines=baseline,baseline_verification=read(EV/'baselines/verification.json'),
    comparison=comparison,parent500_mechanism=read(ROOT/'randomk500/results.json')['mechanism']['RandomK'],
    evaluation_protocol='Frozen native embedding retrieval; FP32; batch64; COCO chunk512; Long-DCI reconstructed',
    limitations=['Single seed; no statistical significance claim','500->3651 comparison changes training budget',
                 'Clean/Full3651 are existing method references, not paired RandomK ablations',
                 'Long-DCI is reconstructed; Flickr30k uses the fixed 1000-image/5000-caption manifest'])
with (DEST/'results.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps([x for x in comparison if x['metric']=='R@1'],indent=2))
