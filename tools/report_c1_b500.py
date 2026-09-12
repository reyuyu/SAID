"""Summarize the explicitly authorized B250-to-B500 extension."""
import hashlib,json,statistics,subprocess
from pathlib import Path

ROOT=Path('/root/SAID-c1-micro-ablation-v01')
RUN=ROOT/'runs_salu/c1_micro_B500'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
ev=RUN/'evaluation'
c=read(ev/'B500_coco.json')['canonical']['C1_VWarm@500']
u=read(ev/'B500_urban1k.json')
assert 'sharegpt4v1k' not in c, 'unexpected final dataset evaluation'
ck=RUN/'c1_C1_text_conditional_reconstruction_step000500.pt'
ck_sha=sha(ck)
assert c['checkpoint_sha256']==ck_sha and u['checkpoint_sha256']==ck_sha
summary=read(RUN/'run_summary.json');config=read(RUN/'config.json')
assert summary['completed_steps']==500 and summary['lr_horizon_steps']==3651
assert config['variant']=='C1-VWarm'
rows=[json.loads(s[4:]) for s in (RUN/'salu_log.jsonl').read_text().splitlines() if s.startswith('LOG ')]
assert rows[0]['completed_steps']>250 and rows[-1]['completed_steps']==500
assert all(x['rec_visual_alpha']==1 and x['next_batch_index']==x['completed_steps'] for x in rows)
log=Path('/root/c1_micro_B500.log').read_text()
replay=[s for s in log.splitlines() if s.startswith('REPLAY_VERIFIED')]
assert len(replay)==1
old=read('/root/SAID-token-v1/docs/said_token_v1/fix_v2_results.json')['retrieval']
a=read(ROOT/'runs_salu/C1_UN_final_coco.json')['canonical']['C1_UN@500']
au=read(ROOT/'runs_salu/C1_UN_final_urban1k.json')['urban1k']
models={name:{'coco':old[name]['coco'],'urban1k':old[name]['urban1k']} for name in ['S0@500','C1@500']}
models['C1-UN@500']={'coco':a['coco_val2017'],'urban1k':au}
models['C1-VWarm@500']={'coco':c['coco_val2017'],'urban1k':u['urban1k']}
b=models['C1-VWarm@500']['coco'];s=models['S0@500']['coco']
passed=b['image2text_R1']>=.6058 and b['text2image_R1']>=.41236 and (b['image2text_R1']>.6058 or b['text2image_R1']>.41236)
parent=ROOT/'runs_salu/c1_micro_B250/c1_C1_text_conditional_reconstruction_step000250.pt'
parent_summary=read(parent.parent/'run_summary.json')
result={'experiment':'C1-VWarm continued from 250 to 500','final_datasets':['COCO val2017','Urban-1k'],
 'models':models,'deltas_coco_B_minus':{n:{k:b[k]-v['coco'][k] for k in b} for n,v in models.items() if n!='C1-VWarm@500'},
 'gate':{'status':'PROMISING_AT_500' if passed else 'FAIL','pass':passed,'rule':'I2T_R1>=.6058 and T2I_R1>=.41236, at least one strictly greater'},
 'parent_checkpoint_sha256':sha(parent),'checkpoint_sha256':ck_sha,'implementation_sha':config['git_head'],
 'resume_evidence':replay[0],'new_optimizer_updates':250,'candidate_total_optimizer_updates':500,
 'candidate_sample_presentations':512000,'matching_candidates':1024,'lr_horizon':3651,
 'continuation_summary':summary,'parent_250_summary':parent_summary,'logged_first_update':rows[0],'logged_last_update':rows[-1],
 'tests':'3 existing micro-ablation tests PASS; CLI syntax PASS; actual sample/caption replay and reference fingerprint PASS',
 'limitations':['checkpoint did not save full model RNG; bitwise full training equivalence not claimed',
 'single seed; point metrics do not establish statistical superiority',
 'prior A/B selection used ShareGPT4V fixed1K, not the requested COCO 1K with five captions; B500 is separately user-authorized and does not rely on that selection'],
 'STOP':True}
dest=ROOT/'docs';(dest/'c1_vwarm_500_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
lines=['# C1-VWarm：250→500 步追加实验','',
 '按用户追加授权，从既有B@250续训250次更新，B累计500步。原A/B/选型续训共750步，本次后探索累计1000次新更新。只报告COCO和Urban-1k原生CLS/EOS，其他基线结果复用。','',
 'B保持旧C1的r0=normalize(v)*m_U，decoder输入前向不变，只对rec回传到视觉的梯度乘alpha。s≤100为0、100<s<200为(s−100)/100、s≥200为1；本次251–500全部为1。','',
 '恢复学生、mask、decoder和三个optimizer；加载父运行冻结reference并验证指纹。重放前250批，在第一条新更新前核对rank0完整sample/caption流SHA；epoch=0，next_batch=250，next_update=251。其他rank未独立保存流SHA，旧checkpoint缺少完整模型RNG，不声称逐位一致。','',
 '## 检索结果（百分数）','']
for dataset in ['coco','urban1k']:
    lines+=['### '+dataset,'','| 模型 | I2T R@1/5/10 | T2I R@1/5/10 |','|---|---|---|']
    for n,m in models.items():
        v=m[dataset]
        vals=lambda direction:[v[direction+'_R'+str(k)] if dataset=='coco' else v[direction]['R'+str(k)] for k in (1,5,10)]
        lines.append('| '+n+' | '+' / '.join(f'{x*100:.3f}' for x in vals('image2text'))+' | '+' / '.join(f'{x*100:.3f}' for x in vals('text2image'))+' |')
    lines+=['']
lines+=['## 晋级门与预算','',f"原始精度COCO R@1：I2T={b['image2text_R1']!r}，T2I={b['text2image_R1']!r}。门结果：**{result['gate']['status']}**。相对S0分别为{b['image2text_R1']-s['image2text_R1']:+.8f}、{b['text2image_R1']-s['text2image_R1']:+.8f}（原始比例差）。单seed不证明统计显著优势。",'',
 f"本次新增256000对样本呈现，B累计512000对。4×256、accumulation=1、horizon3651。续训墙钟{summary['wall_sec']:.1f}秒（包含数据重放），每步记录均值{summary['mean_sec_per_step']:.3f}秒，峰值{summary['peak_memory_gb']:.2f}GiB/卡。",'',
 '3项原有微消融测试通过、CLI语法通过，本次实际恢复流与reference校验通过。原三项测试不覆盖完整生产DDP梯度等价，不扩张其结论。','',
 '历史说明：先前A/B开发选型实际使用ShareGPT4V固定1K，不能称为COCO 1000图×5caption开发集；本次B500来自用户独立追加授权。','',
 '实现SHA：`'+config['git_head']+'`。父checkpoint SHA256：`'+result['parent_checkpoint_sha256']+'`；B500 SHA256：`'+ck_sha+'`。','',
 '完成后STOP。未训练新变体、未延长到3epoch；本次未评估ShareGPT4V。']
(dest/'c1_vwarm_500_report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'gate':result['gate'],'coco':b,'urban':u['urban1k']},indent=2))
