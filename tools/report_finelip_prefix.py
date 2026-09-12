"""Compile verified FP0 outputs and reused baselines; never manufacture missing evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def read(path): return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    run=Path(a.run); root=Path(__file__).resolve().parents[1]; dest=root/'docs/finelip_prefix'
    config=read(run/'config.json'); summary=read(run/'run_summary.json')
    e=config['updates_per_epoch']; ev=run/'evaluation'
    steps=[0,1000,e,2*e]
    curve=[]
    for step in steps:
        label='FP0_update%d'%step
        canonical=read(ev/(label+'_canonical.json'))['canonical'][label]
        urban=read(ev/(label+'_urban1k.json'))
        provenance=read(ev/(label+'_provenance.json'))
        if canonical['checkpoint_sha256']!=provenance['student_sha256'] or urban['checkpoint_sha256']!=provenance['student_sha256']:
            raise ValueError('evaluation/checkpoint hash mismatch')
        curve.append({'label':label,'coco':canonical['coco_val2017'],'urban1k':urban['urban1k'],
                      'sharegpt4v1k':canonical['sharegpt4v1k'],'provenance':provenance,
                      'diagnostics':read(ev/(label+'_diagnostics.json'))})
    prior=read('/root/SAID-token-v1/docs/said_token_v1/fix_v2_results.json')
    baselines={k:v for k,v in prior['retrieval'].items() if k in ['Initial','S0@500','T1_fix_v2@500']}
    am=Path('/root/SAID-token-adaptive-v01/runs_salu/said_token_adaptive_v01/am_step500_20260912/evaluation')
    am_c=read(am/'T1_fix_v2_step500_canonical.json')['canonical']['T1_fix_v2_step500']
    baselines['AM@500']={'coco':am_c['coco_val2017'],'urban1k':read(am/'T1_fix_v2_step500_urban1k.json')['urban1k'],
                         'source_path':str(am),'checkpoint_sha256':am_c['checkpoint_sha256']}
    metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    tails=[x for x in metrics if x['next_batch_index']==config['loader_batches']]
    if len(tails)!=2 or any(x['actual_group_size']!=2 for x in tails): raise ValueError('missing exact epoch tails')
    initial_delta={k:curve[0]['coco'][k]-baselines['Initial']['coco'][k] for k in curve[0]['coco']}
    result={'experiment':'FineLIP-Prefix FP0','description':'依据FineLIP方法与公开技术说明独立实现的前缀监督基线',
            'config':config,'run_summary':summary,'retrieval_curve':curve,'reused_baselines':baselines,
            'initial_coco_delta_vs_historical':initial_delta,'epoch_tail_records':tails,
            'implementation_sha':config['git_sha'],
            'report_parent_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'validation':read(dest/'validation.json'),'NOT_RUN':['UPSTREAM_RUNTIME_PARITY','full formal worker-resume experiment','six-epoch training'],
            'license_status':['LICENSE_NOT_IDENTIFIED','REUSE_PERMISSION_UNCONFIRMED'],
            'no_upstream_code_runtime_dependency':True,'UNSAID_SEMANTICS':'NOT ESTABLISHED',
            'training_peak_memory_gib':max(x['peak_memory_gib'] for x in metrics),
            'median_logged_update_wall_seconds':statistics.median(x['update_wall_seconds'] for x in metrics if x['optimizer_step']>20),
            'paper_reference':{'url':'https://arxiv.org/html/2504.01916v1','version':'arXiv v1, Table 1',
                               'FineLIP_B16_Urban1k_I2T':[.907,.983,.995],'FineLIP_B16_Urban1k_T2I':[.893,.975,.987],
                               'training_epochs':6,'comparison':'external paper reference, not same training budget or prefix protocol'}}
    dest.mkdir(exist_ok=True)
    (dest/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    text=['# FineLIP-Prefix FP0：两 epoch 实验报告','',result['description']+'。','',
          '代码独立实现，不复制或动态导入外部 FineLIP checkout。上游参考固定为 `'+config['upstream_sha']+'`；许可证未识别、直接复用授权未确认；UPSTREAM_RUNTIME_PARITY: NOT RUN。','',
          '实际训练：4×32 microbatch，128 候选，累积4；一组平均四个独立 microbatch 的 hinge SUM，不是512候选。数据样本数 %d，每epoch %d microbatch、%d updates；6epoch horizon=%d，本次在%d updates结束。'%(config['dataset_size'],config['loader_batches'],e,config['lr_horizon_updates'],summary['optimizer_step']),'',
          '保存 init/1000/epoch1/epoch2，逐一严格导出317个学生 tensors，主评估只用原生 CLS/EOS。两次epoch尾部均实际累积2个microbatch、%d对样本，已flush。'%(tails[0]['group_samples']),'',
          '相对上游的明确区别：随机前k句、共同初始化与seed、bf16主干/FP32核心、重复image和有效前缀过滤、EOS边界（包含SOT和内容ID=0）、optimizer-step scheduler、每epoch shuffle、实际累积尾部、非有限值硬失败。','',
          '## 原生检索（百分数）','']
    ordered=list(baselines.items())+[(x['label'],x) for x in curve]
    for dataset in ['coco','urban1k']:
        text+=['### '+('COCO val2017：5000图/25000文本' if dataset=='coco' else 'Urban-1k：1000图/1000文本'),'',
               '| 模型 | I2T R@1/5/10 | T2I R@1/5/10 |','|---|---|---|']
        for label,row in ordered:
            d=row[dataset]
            ir=[d['image2text_R'+str(k)] for k in (1,5,10)] if dataset=='coco' else [d['image2text']['R'+str(k)] for k in (1,5,10)]
            tr=[d['text2image_R'+str(k)] for k in (1,5,10)] if dataset=='coco' else [d['text2image']['R'+str(k)] for k in (1,5,10)]
            text.append('| %s | %s | %s |'%(label,' / '.join('%.2f'%(v*100) for v in ir),' / '.join('%.2f'%(v*100) for v in tr)))
        text.append('')
    text+=['ShareGPT4V 固定1K 的 first_sentence/fixed_sparse/full_dense 各自结果完整保存在 results.json 的 sharegpt4v1k 字段，不混入 COCO 或 Urban-1k。','',
           '## 预算与性能','',
           'FP0@1000呈现512000对，接近旧S0/T1/AM@500的样本呈现量；但optimizer updates为1000对500，候选池为128对1024，计算成本不同，不能称完全等预算。','',
           '两epoch呈现 %d 对（含DistributedSampler每epoch为对齐rank补齐的3条重复），训练墙钟 %.1f 秒，累计四卡计算 %.3f GPU小时；峰值显存 %.2f GiB，日志更新耗时中位数 %.3f 秒。'%(summary['sample_presentations'],summary['wall_seconds'],summary['gpu_compute_hours'],result['training_peak_memory_gib'],result['median_logged_update_wall_seconds']),'',
           '## 固定64前缀诊断','',
           '| update | image scale | text scale | image slot cos | text slot cos | native I2T/T2I R1 | finegrain I2T/T2I R1 |',
           '|---|---|---|---|---|---|---|']
    for row in curve:
        d=row['diagnostics'];text.append('| %d | %.5f | %.5f | %.6f | %.6f | %.3f / %.3f | %.3f / %.3f |'%(row['provenance']['optimizer_step'],d['image_scale'],d['text_scale'],d['image_slot_cos'],d['text_slot_cos'],d['native_cls_eos']['I2T_R1'],d['native_cls_eos']['T2I_R1'],d['finegrain']['I2T_R1'],d['finegrain']['T2I_R1']))
    text+=['','此固定训练cohort只用于解释局部与全局行为，不是独立验证集主指标，不能替代canonical。聚合权重熵和固定manifest另存。','',
           '## 来源与验收','',
           '6项必要测试通过，涵盖公式、负cosine、梯度符号/平移、EOS/重复样本、两rank真实累积与尾部AdamW更新、严格状态恢复。共享初始化GPU验收确认native接口一致、40-token集合和全主干backward。测试详情见 validation.json。','',
           '论文外部参考：[FineLIP, arXiv v1 表1](https://arxiv.org/html/2504.01916v1)：B/16 Urban-1k I2T=90.7/98.3/99.5，T2I=89.3/97.5/98.7。论文使用6epoch长caption训练；本次是2epoch随机前缀、项目共同初始化。仅列外部参考，不当成同框架对照；未用FineLIP*混合细粒度推理结果代替原生CLS。','',
           '实现SHA：`'+config['git_sha']+'`。各checkpoint与裸学生SHA见results.json provenance。','',
           'NOT RUN：官方运行时数值对照；完整正式worker恢复续训（已有恢复单测）；六epoch及任何新变体。未证明Said/Unsaid语义分离。完成本次两epoch与评估后STOP。']
    (dest/'report.md').write_text('\n'.join(text)+'\n')
    (dest/'fixed64_manifest.json').write_text((ev/'fixed64_manifest.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,dataset in zip(axes,['coco','urban1k']):
        for direction,color in [('image2text','#176b98'),('text2image','#bb5b29')]:
            vals=[(r[dataset][direction+'_R1'] if dataset=='coco' else r[dataset][direction]['R1'])*100 for r in curve]
            ax.plot(steps,vals,'o-',color=color,label=direction+' FP0')
            base=baselines['S0@500'][dataset]
            baseline=(base[direction+'_R1'] if dataset=='coco' else base[direction]['R1'])*100
            ax.axhline(baseline,color=color,linestyle='--',alpha=.55,label=direction+' S0@500')
        ax.set(title=dataset, xlabel='FP0 optimizer updates',ylabel='Native CLS/EOS R@1 (%)')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('FP0 prefix baseline; dashed S0 reference uses a different training budget',fontsize=10)
    fig.savefig(dest/'retrieval_curve.png',dpi=180)
    plt.close(fig)
    print(dest/'report.md')


if __name__=='__main__':main()
