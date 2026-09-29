import json,pathlib,datetime,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'status.json').read_text())
p=ROOT/'run/salu_log.jsonl'
if p.exists():
 rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 if rows:
  last=rows[-1];s['latest_update']={k:last.get(k) for k in ['completed_steps','epoch','step_in_epoch','loss_total','loss_suffix_global','lr','synchronized_step_seconds','peak_reserved_mb']};s['new_logged_updates']=len(rows);s['recent_mean_step_seconds']=statistics.mean(r['synchronized_step_seconds'] for r in rows[-100:]);s['log_age_seconds']=datetime.datetime.now().timestamp()-p.stat().st_mtime
else:s['latest_update']={'completed_steps':500,'note':'checkpoint restoration / replay of consumed batches; no new optimizer update logged yet'}
p=ROOT/'reports/evaluation_progress.json'
if p.exists():
 x=json.loads(p.read_text());s['evaluation_progress']={'completed':len(x['completed']),'total':x['total'],'protocols':[v['protocol'] for v in x['completed']]}
print(json.dumps(s,indent=2,ensure_ascii=False))
