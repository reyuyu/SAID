"""Read-only comparison of the requested checkpoint against its existing trajectory."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/root/lk_projects/SAID-nest-clip-v1')
RUN=ROOT/'randomk2000'
DEST=Path('/root/lk_projects/SAID/experiments/nest_clip_v1/randomk2000')
protocols=('coco','urban','flickr_test1k','docci')
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pre=read(RUN/'evidence/preflight.json');check=read(RUN/'export-check.json')
assert check['passed'] and check['optimizer_steps']==[2000]
assert check['image_max_abs']==check['text_max_abs']==0
assert check['checkpoint_sha256']==pre['checkpoint_sha256']
executions={k:read(RUN/f'evidence/{k}.execution.json') for k in ('export2000','verify-export2000',*protocols)}
assert all(x['exit_code']==0 for x in executions.values())
raw={};sources={}
for step in (500,2000,3651):
    raw[str(step)]={};sources[str(step)]={}
    for p in protocols:
        if step==2000:
            path=RUN/f'evaluation/{p}_native.json' if p in ('coco','urban') else RUN/f'evaluation/{p}/{p}.json'
        elif step==500 and p in ('coco','urban'):
            path=ROOT/f'randomk500/A3-RandomK/{p}_native.json'
        else:
            folder=ROOT/f'randomk3epoch/evaluation/step{step}'
            path=folder/f'{p}_native.json' if p in ('coco','urban') else folder/p/f'{p}.json'
        raw[str(step)][p]=read(path);sources[str(step)][p]=dict(path=str(path),sha256=digest(path))
        if step==2000:assert raw[str(step)][p]['checkpoint_sha256']==check['bare_sha256']
def recall(result,p,d,k):
    key='image2text' if d=='I2T' else 'text2image'
    if p=='coco':return result[key+f'_R{k}']
    if p=='urban':return result[key][f'R{k}']
    return result['metrics'][f'R@{k}'][d]
rows=[]
for p in protocols:
    for d in ('I2T','T2I'):
        for k in (1,5,10):
            v={str(s):recall(raw[str(s)][p],p,d,k) for s in (500,2000,3651)}
            assert all(0<=x<=1 for x in v.values())
            rows.append(dict(protocol=p,direction=d,metric=f'R@{k}',raw=v,
                             percent={s:100*x for s,x in v.items()},
                             delta_2000_vs500_pp=100*(v['2000']-v['500']),
                             delta_2000_vs3651_pp=100*(v['2000']-v['3651'])))
result=dict(model='A3-RandomK',evaluated_step=2000,new_training=False,preflight=pre,
            export_check=check,executions=executions,raw_evaluations=raw,sources=sources,
            comparison=rows,protocols=protocols,excluded_protocols=['dci','long_dci'],
            policy='Do not run DCI/Long-DCI in this and future tasks unless the user changes scope',
            limitations=['Same seed=0 trajectory, different checkpoint budgets',
                         'User-requested post-hoc checkpoint inspection; no statistical significance claim'])
(RUN/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
names={'coco':'COCO','urban':'Urban-1k','flickr_test1k':'Flickr30k test1k','docci':'DOCCI'}
lines=['A3-RandomK step2000：四项原生检索评测','',
       '只评估用户指定的同一训练轨迹step2000检查点，没有新训练、调参或自动追加其他检查点。根据最新要求，本次及后续跳过DCI full与Long-DCI，历史结果保留。','',
       '训练版本：`'+pre['training_code']+'`；本次执行版本：`'+executions['export2000']['git_head']+'`。原始scheduler horizon仍为3651，checkpoint及optimizer步数均为2000。严格导出/native图像文本一致性核验通过，最大差均为0。','',
       '下表单位为%，差值为2000减比较点的百分点（pp）。500/3651复用已完成的原始结果，不重新评测。','']
def table(headers,values):
    lines.append('| '+' | '.join(headers)+' |');lines.append('|'+'|'.join(['---']*len(headers))+'|')
    lines.extend('| '+' | '.join(map(str,x))+' |' for x in values);lines.append('')
table(['数据集','方向','指标','500','2000','3651','2000−500 pp','2000−3651 pp'],[
    [names[x['protocol']],x['direction'],x['metric'],*[f"{x['percent'][str(s)]:.3f}" for s in (500,2000,3651)],f"{x['delta_2000_vs500_pp']:+.3f}",f"{x['delta_2000_vs3651_pp']:+.3f}"] for x in rows])
for key,label in [('delta_2000_vs500_pp','500'),('delta_2000_vs3651_pp','3651')]:
    lines+= [f"24项指标相对{label}：{sum(x[key]>0 for x in rows)}项上升、{sum(x[key]<0 for x in rows)}项下降、{sum(x[key]==0 for x in rows)}项持平。",'']
lines+=['协议保持不变：COCO canonical 5000图/25000文本，similarity_chunk512；Urban 1000/1000；Flickr test1k 1000/5000；DOCCI test 5000/5000。图像batch64，COCO/Urban沿用现有文本编码和排序，扩展评测复用冻结实现、FP32与OMP/MKL=4；均为归一化native embedding内积，无mask/融合/rerank。','',
        '训练检查点SHA256：`'+check['checkpoint_sha256']+'`。裸学生SHA256：`'+check['bare_sha256']+'`。大权重保留服务器；原始JSON、命令、退出码、评测器SHA256和完整对照见 [results.json](results.json) 与 [evidence](evidence)。','']
table(['阶段','退出码','耗时秒'],[[k,x['exit_code'],f"{x['wall_seconds']:.3f}"] for k,x in executions.items()])
lines+=['这是同一seed=0轨迹的事后检查点比较，不能据小幅差值宣称统计显著；没有根据这些结果自动选点继续训练。复现命令见 [run.sh](run.sh)，后续范围见 [评测规范](../EVALUATION_POLICY.md)。']
(RUN/'REPORT.md').write_text('\n'.join(lines).rstrip()+'\n')
manifest=[]
def copy(p,target):
    dst=DEST/target;dst.parent.mkdir(parents=True,exist_ok=True)
    assert not dst.exists(),dst
    shutil.copyfile(p,dst);manifest.append(dict(path=target,server_source=str(p),sha256=digest(dst),bytes=dst.stat().st_size))
for p in sorted((RUN/'evidence').glob('*')):
    if p.is_file() and p.suffix in ('.py','.json','.txt'):copy(p,'evidence/'+p.name)
for p in sorted((RUN/'evaluation').rglob('*.json')):copy(p,'evaluation/'+str(p.relative_to(RUN/'evaluation')))
for step in ('500','3651'):
    for protocol,src in sources[step].items():copy(Path(src['path']),f'references/step{step}/{protocol}.json')
for name in ('export-check.json','results.json','REPORT.md'):copy(RUN/name,name)
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(DEST/'.gitattributes').write_text('evidence/*.txt -whitespace\n')
print(json.dumps([x for x in rows if x['metric']=='R@1'],indent=2))
