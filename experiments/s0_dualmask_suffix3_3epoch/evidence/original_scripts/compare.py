import csv,json,math,pathlib,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
DOCS=pathlib.Path('/root/lk_projects/SAID-publish-suffix3')

for filename,expected in json.loads((ROOT/'validation/reference_files.json').read_text()).items():
 assert hashlib.sha256(pathlib.Path(filename).read_bytes()).hexdigest()==expected,filename

done=json.loads((ROOT/'reports/evaluation_complete.json').read_text());assert done['status']=='PASS' and done['evaluations']==6
base=json.loads(pathlib.Path('/root/lk_projects/SAID-reproduction/evaluation/comparison.json').read_text())
refs={'suffix3_500':json.loads(pathlib.Path('/root/lk_projects/SAID-tuning-500/reports/suffix3_extended_comparison.json').read_text())['metrics']}
for name in ['clean3651','full3651']:refs[name]={r['protocol']:r['replica'] for r in base['comparisons'] if r['model']==name}
historical=json.loads((DOCS/'experiments/results.json').read_text())
for name,experiment in [('historical_clean3651','S0-DualMask-Clean-v0.1'),('historical_full3651','S0-DualMask-Full-v0.1')]:refs[name]={r['protocol']:r['metrics'] for e in historical['experiments'] if e['experiment']==experiment and e['steps']==3651 for r in e['results']}
protocols=['coco','urban','flickr_test1k','docci','dci','long_dci'];current={};rows=[]
for p in protocols:
 raw=json.loads((ROOT/'evaluation'/p/(p+'.json')).read_text());assert raw['checkpoint_sha256']==done['checkpoint_sha256']
 if p=='coco':m={f'R@{k}':{'I2T':raw['metrics'][f'image2text_R{k}'],'T2I':raw['metrics'][f'text2image_R{k}']} for k in [1,5,10]}
 elif p=='urban':m={f'R@{k}':{'I2T':raw['urban1k']['image2text'][f'R{k}'],'T2I':raw['urban1k']['text2image'][f'R{k}']} for k in [1,5,10]}
 else:m=raw['metrics']
 current[p]=m
 for k in [1,5,10]:
  for d in ['I2T','T2I']:
   v=m[f'R@{k}'][d];assert math.isfinite(v) and 0<=v<=1
   row={'protocol':p,'metric':f'R@{k}','direction':d,'suffix3_3651_percent':v*100}
   for name,metrics in refs.items():row[name+'_percent']=metrics[p][f'R@{k}'][d]*100;row['delta_vs_'+name+'_pp']=(v-metrics[p][f'R@{k}'][d])*100
   rows.append(row)
summary={}
for name in refs:
 v=[r['delta_vs_'+name+'_pp'] for r in rows if r['metric']=='R@1'];summary[name]={'R1_improved':sum(x>1e-5 for x in v),'R1_decreased':sum(x< -1e-5 for x in v),'R1_tied':sum(abs(x)<=1e-5 for x in v),'mean_R1_delta_pp':sum(v)/len(v)}
result={'status':'COMPLETE','model':'S0-DualMask-Suffix3-U0','completed_steps':3651,'epochs':3,'metrics':current,'comparators':refs,'summary':summary,'rows':rows,'export':json.loads((ROOT/'exports/export.json').read_text()),'note':'same-machine Clean/Full3651 are the primary equal-budget comparators; own500 is a training-length comparison, and historical A800 results are separately named.'}
(ROOT/'reports/results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
with (ROOT/'reports/comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
labels={'coco':'COCO','urban':'Urban-1k','flickr_test1k':'Flickr30k test1K','docci':'DOCCI','dci':'DCI','long_dci':'Long-DCI重建版'}
lines=['# 3/0 完整3 epoch结果','','从已校验的3/0@500完整状态恢复，新增3151步，总计3651步。保持lambda_suffix=3、lambda_u_sparse=0、4×256、seed0、LR=1e-6/1e-3/1e-4和原3651步调度，不重新warmup。训练、严格导出与六项评测均成功。','','## R@1对照','','单位%，每格为图→文 / 文→图。Clean/Full为此前在同一机器完成的3651步复现，500步列只用于观察训练长度变化。','','| 数据集 | 3/0@500 | 3/0@3651 | Clean@3651 | Full@3651 | Δ vs Clean（百分点） |','|---|---:|---:|---:|---:|---:|']
for p in protocols:
 fmt=lambda m:' / '.join(f"{m[p]['R@1'][d]*100:.3f}" for d in ['I2T','T2I'])
 delta=' / '.join(f"{(current[p]['R@1'][d]-refs['clean3651'][p]['R@1'][d])*100:+.3f}" for d in ['I2T','T2I'])
 lines.append(f"| {labels[p]} | {fmt(refs['suffix3_500'])} | {fmt(current)} | {fmt(refs['clean3651'])} | {fmt(refs['full3651'])} | {delta} |")
lines+=['','## 本模型全部指标','','| 协议 | 方向 | R@1 | R@5 | R@10 |','|---|---|---:|---:|---:|']
for p in protocols:
 for d in ['I2T','T2I']:lines.append(f"| {labels[p]} | {d} | "+' | '.join(f"{current[p][f'R@{k}'][d]*100:.3f}" for k in [1,5,10])+' |')
lines+=['','## 证据与解释范围','','- [完整CSV及原历史值](comparison.csv) · [结果JSON](results.json) · [训练前恢复验证](../validation/preflight.json) · [严格导出](../exports/export.json)。','- [裸学生](../exports/bare_student_step3651.pt) · [完整训练checkpoint](../run/s0_dual_mask_suffix_masked_step003651.pt)。','- 原500步目录未覆盖。新训练日志只包含501–3651；合并记录位于training_steps_000001_003651.jsonl。','- 使用原生FP32检索、冻结COCO/Urban函数和原扩展评测脚本；Long-DCI保留重建版口径。','- 本次直接验证500步选定配置延长训练后的实际结果，不承诺全面领先，不据单seed、小幅分差宣称统计显著。','- 续训使用固定873b43a实现：重放/跳过已消费batch，续训摘要从第501步开始记录；不将其描述为独立的全历史内容级回放校验。']
(ROOT/'reports/FINAL_REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2),flush=True)
