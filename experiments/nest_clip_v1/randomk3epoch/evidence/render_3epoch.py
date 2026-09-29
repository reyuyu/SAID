"""Render final measured six-dataset results and continuation evidence."""
import json
from pathlib import Path
ROOT=Path('/root/lk_projects/SAID-nest-clip-v1/randomk3epoch')
r=json.loads((ROOT/'results.json').read_text());a=r['audit'];lines=[]
names={'coco':'COCO canonical','urban':'Urban-1k','flickr_test1k':'Flickr30k test1k','docci':'DOCCI test','dci':'DCI full','long_dci':'Long-DCI重建版'}
def add(s=''):lines.append(s)
def table(headers,rows):
    add('| '+' | '.join(headers)+' |');add('|'+'|'.join(['---']*len(headers))+'|')
    for row in rows:add('| '+' | '.join(map(str,row))+' |')
    add()
add('A3-RandomK：完整3epoch与六项原生检索')
add()
add('从已完成的step500严格接续新增3151更新，累计完成3651更新（3×1217）。完整scheduler horizon始终为3651，不是重新训练3651步，也没有重启warmup。仅评最终step3651；为观察同一方法的预算变化，补齐step500的四项扩展评测，复用其已有COCO/Urban结果。未改RandomK、包含权重、网络或损失，未调参或挑选中间最好checkpoint。')
add()
r1=[x for x in r['comparison'] if x['metric']=='R@1']
if all(x['delta_pp']['RandomK500']>0 for x in r1):
    add('**六个数据集的双向R@1均较step500提高。** 这说明本次A3-RandomK轨迹在完整3epoch预算下取得了更好的原生检索结果；与同预算Clean/Full仍需逐项比较，不能概括为全面领先。')
else:
    add(f"相对step500，12个R@1方向有{sum(x['delta_pp']['RandomK500']>0 for x in r1)}项上升、{sum(x['delta_pp']['RandomK500']<0 for x in r1)}项下降；具体结果如下。")
add()
add('续训代码：`'+r['code_commit']+'`。起点检查点SHA256：`'+a['config']['parent_checkpoint_sha256']+'`。模型、目标与数据源码哈希保持父版本一致，仅修改trainer的停止点/严格恢复验证和完整epoch结束控制；旧trainer哈希由命令显式固定。见 [run.sh](run.sh)、[配置](training/config.json)、[恢复核验](evidence/restore-check.json) 和 [测试](evidence/tests.txt)。')
add()
add('单机4张A100 80GB，256/rank、accumulation=1、seed=0；完整1,245,901训练条目，DistributedSampler padding=3/epoch，末批每rank180（全局720），其他满批全局1024。K仍从可见F中均匀抽取1…n−1，由(seed=0,epoch,原sample_id)派生，日志O/E分别对应变长P/R。')
add()
add('R@1结果（百分比；差值为3651−500，单位pp）：')
add()
table(['协议','方向','RandomK@500','RandomK@3651','Δ pp','Clean@3651参考','Full@3651参考'],[
    [names[x['protocol']],x['direction'],f"{x['percent']['RandomK500']:.3f}",f"{x['percent']['RandomK3651']:.3f}",f"{x['delta_pp']['RandomK500']:+.3f}",f"{x['percent']['clean3651']:.3f}",f"{x['percent']['full3651']:.3f}"]
    for x in r['comparison'] if x['metric']=='R@1'])
add('完整双向R@1/5/10：')
add()
table(['协议','方向','指标','500 (%)','3651 (%)','3651−500 pp','Clean3651 (%)','Full3651 (%)'],[
    [names[x['protocol']],x['direction'],x['metric'],f"{x['percent']['RandomK500']:.3f}",f"{x['percent']['RandomK3651']:.3f}",f"{x['delta_pp']['RandomK500']:+.3f}",f"{x['percent']['clean3651']:.3f}",f"{x['percent']['full3651']:.3f}"] for x in r['comparison']])
delta=[x['delta_pp']['RandomK500'] for x in r['comparison']]
add(f"共36项指标：相对step500，上升{sum(x>0 for x in delta)}项、下降{sum(x<0 for x in delta)}项、持平{sum(x==0 for x in delta)}项。训练预算增加是这次轨迹比较的变量，不能据此单独证明包含项或随机K有效。")
add()
add('协议固定：COCO 5000图/25000文本，batch64、similarity_chunk512；Urban 1000/1000，沿用原一次性文本编码；Flickr30k test1k 1000/5000，DOCCI test 5000/5000，DCI full 7805/7805，Long-DCI重建版7602/7602。扩展评测直接复用冻结eval_extended_real.py，FP32原生归一化向量内积、batch64、历史OMP/MKL=4与原排序，无mask/融合/rerank。Long-DCI使用重建协议；Flickr范围以1000图/5000文本清单为准。')
add()
add('Clean/Full参考均来自已核验的本机3651步权重和既有六项评测，未重训。Full导出文件名包含step500是旧命名，实际checkpoint元信息及optimizer步数为3651，按哈希识别。[参考核验](evidence/baselines/verification.json)。它们是不同方法的横向参考，不是本轮单因素消融。')
add()
add('续训验收：恢复后的模型、AdamW状态和四rank Python/NumPy/CPU/CUDA RNG与父检查点逐项相同；第一新增更新step501使用s=500，之后500→3651连续无遗漏。四rank各新增3151更新，累计3651，末尾参数相对rank0最大差为0。所有记录的loss/梯度有限，包含权重按原全局更新数计算，未重新warmup。')
add()
table(['日志epoch','更新数','样本行数(含padding)','局部有效比例','K=1比例(有效样本)','回退原因'],[
    [e,s['updates'],s['samples'],f"{100*s['valid_ratio']:.5f}%",f"{100*s['k1_ratio']:.5f}%",json.dumps({k:v for k,v in s['reasons'].items() if k!='valid'},ensure_ascii=False)] for e,s in a['epoch_stats'].items()])
add(f"三epoch合计{a['samples_total']:,}个样本行。全部sample_id逐项匹配DistributedSampler，全部K按对应epoch重新派生并重放通过；并不要求每个样本跨epoch的K必然改变。每epoch最后一批实际180/rank，全局720候选，日志已核验。")
add()
table(['step','epoch','各rank实际batch','F候选','有效局部数','包含权重','backbone LR','mask LR'],[
    [k,x['epoch'],'/'.join(str(h['batch']) for h in x['rank_health']),x['F_candidates'],x['valid_global'],x['inc_weight'],f"{x['lr_backbone']:.10g}",f"{x['lr_mask']:.10g}"] for k,x in a['epoch_boundary_steps'].items()])
add('最后50步的逐步算术均值如下。3651的窗口包含最终720候选的尾批；common_loss=total_loss−inc_weight×inc。')
add()
keys=['loss','common_loss','F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i','inc','inc_weight','hard_inclusion_violation','F_keep_ratio','O_keep_ratio','E_keep_ratio','F_all_open','O_all_open','E_all_open','F_all_closed','O_all_closed','E_all_closed','oe_iou','valid_global']
old=r['parent500_mechanism']['last50']
table(['量(O=P,E=R)','step500最后50步','step3651最后50步'],[[k,f"{old[k]:.6f}",f"{a['last50'][k]:.6f}"] for k in keys])
add('hard包含违例是逐特征维度的平均违例比例，全开/全关是样本比例；P/R IoU是hard-mask交并比。这些训练batch日志不代替独立检索质量评价。')
add()
table(['rank/GPU','PID','GPU UUID','本次循环耗时秒','峰值allocated GiB','峰值reserved GiB'],[
    [h['rank'],i['pid'],i['uuid'],f"{h['seconds']:.3f}",f"{h['peak_allocated_gib']:.3f}",f"{h['peak_reserved_gib']:.3f}"] for i,h in zip(a['config']['ranks'],a['acceptance']['ranks'])])
add('循环耗时含读取/跳过前500批恢复数据位置、worker启动、3151次同步更新、保存和末尾一致性核验。完整进程耗时另含模型构造。大checkpoint保存了续训起点500、每100步以及最终3651；原先0–500的检查点仍在旧目录。')
add()
table(['阶段','退出码','进程耗时秒'],[[name,x['exit_code'],f"{x['wall_seconds']:.3f}"] for name,x in sorted(r['executions'].items(),key=lambda p:p[1]['started_utc'])])
add('本次训练诊断标记：`'+json.dumps(a['diagnostics'],ensure_ascii=False)+'`。原step500任务关闭worker时的析构异常已在旧报告披露；父权重、optimizer和RNG完整核验后才用于此次续训。')
add()
add('最终step3651训练检查点SHA256：`'+a['checkpoints'][-1]['sha256']+'`。裸学生SHA256：`'+r['export']['bare_sha256']+'`。严格导出核验通过，optimizer step=3651，native图像/文本embedding与训练模块接口最大差均为0。所有保存点SHA256见results.json。')
add()
add('完整机器可读结果：[results.json](results.json)。六项原始评测JSON：[evaluation](evaluation)。公开的[轨迹摘要](training/trajectory_summary.jsonl.gz)是明确标注的派生投影，保留全部更新指标、四rank健康信息和样本/视图摘要；原始逐样本n/K数组及全量日志保留服务器，路径/大小/SHA256列在results.json的audit.raw_logs，不将投影冒充原始日志。')
add()
add('限制：单seed，没有显著性检验；500→3651改变了训练预算，Clean/Full改变了方法。数据资产与Long-DCI重建协议的既有限制保留。模型训练在累计3651停止，没有继续延长日程、挑中间最佳结果或追加消融。')
(ROOT/'REPORT.md').write_text('\n'.join(lines).rstrip()+'\n')
