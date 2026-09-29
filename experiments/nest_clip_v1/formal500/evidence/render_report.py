"""Render tables from measured formal500_results.json; no model execution."""
import json
from pathlib import Path

REPO = Path('/root/lk_projects/SAID')
DEST = REPO/'experiments/nest_clip_v1'
r = json.loads((DEST/'formal500_results.json').read_text())
lines = []


def add(text=''):
    lines.append(text)


def table(headers, rows):
    add('| ' + ' | '.join(headers) + ' |')
    add('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in rows:
        add('| ' + ' | '.join(str(x) for x in row) + ' |')
    add()


def sequence(arm):
    values = r['mechanism'][arm]
    return list(values['milestones'].items()) + [('最后50步均值', values['last50_mean'])]


add('NEST-CLIP v1：A2/A3 500 步固定预算对照')
add()
add('两组均从同一未训练 step-0 独立开始，按 A2→A3 顺序，各完成 **500 次四卡同步更新**；仅评测 step500。训练、导出、严格加载核验和四项原生检索命令均正常返回 0。未调参、未改方法、未使用中间检查点挑选结果，也未追加训练或消融。')
add()
add('**本次结论：A3降低了软/硬包含违例，但没有带来一致的原生检索改善。** COCO六项Recall均下降；Urban-1k仅I2T R@1上升0.200 pp，另3项下降、2项持平。违例降低伴随F选择范围扩大，O/E选择仍有较高重叠。以下结果限于单次seed=0、500步预算。')
add()
add(f"审查及实际运行版本：`{r['reviewed_revision']}`，分支 `codex/nest-clip-v1`。训练与评测源码、配置及 `run.sh` 均保持此版本。原 [5步冒烟报告](REPORT.md) 保留；本报告数值来自 [机器可读结果](formal500_results.json) 和 [本轮证据](formal500/evidence)。")
add()
add('先看 step500 原生检索；Recall 从原始 0–1 数值乘100展示，差值为 **100×(A3−A2)**，单位 pp。')
add()
table(['数据集','方向','指标','A2 (%)','A3 (%)','A3−A2 (pp)'],[
    ['COCO' if v['dataset']=='coco' else 'Urban-1k', v['direction'],v['metric'],
     f"{v['A2']:.3f}",f"{v['A3']:.3f}",f"{v['delta_pp']:+.3f}"] for v in r['recall_comparison_percent']])
delta=r['recall_comparison_percent']
up=sum(x['delta_pp']>0 for x in delta); down=sum(x['delta_pp']<0 for x in delta)
add(f'12项指标中，A3较A2上升 {up} 项、下降 {down} 项、持平 {12-up-down} 项。具体方向以表中结果为准；这是一组 seed=0、500步对照，不能据此宣称统计显著或全面领先。旧 S0/Clean/Full 不作为本轮同预算对照。')
add()
add('两组共同配置：单机4张A100 80GB PCIe、WORLD_SIZE=4、每rank batch=256、accumulation=1、seed=0。完整训练集1,245,901条；每rank sampler长度311,476，padding=3，尾批每rank180；每epoch1217更新、epochs=3、scheduler horizon=3651，stop_updates=500。Backbone LR=1e-6、warmup=200；mask LR=1e-3、无warmup；保持原cosine、BF16编码器/FP32主参数与损失、非重入encoder checkpoint、score_chunk=64。方法差异仅A3包含项；A3在第200步使用0.995、第201步到1，第500步仍为1；A2始终为0。')
add()
add('共同初始化 SHA256：`'+r['preflight']['initialization_sha256']+'`。原始CLIP及MaskNetwork来源沿用审查版本，未使用smoke或A2权重初始化A3。两组step0模型/optimizer均与共同初始化一致，各rank初始RNG一致，全部500步×4rank样本/文本/token摘要逐项一致。')
add()
table(['检查','A2','A3'],[
 ['四rank完成更新数','500 / 500 / 500 / 500','500 / 500 / 500 / 500'],
 ['日志连续范围','1–500，无遗漏/重复','1–500，无遗漏/重复'],
 ['scheduler_horizon / stop_updates','3651 / 500','3651 / 500'],
 ['loss与梯度有限','通过','通过'],
 ['末尾相对rank0最大参数差','0','0'],
 ['保存更新点','0,100,200,300,400,500','0,100,200,300,400,500'],
 ['strict export / optimizer step','通过 / 500','通过 / 500'],
 ['native image/text embedding最大差','0 / 0','0 / 0']])
add('原生协议固定为COCO canonical 5000图/25000文本（原注释25014条，每图取原顺序前5条），similarity_chunk=512；Urban-1k为1000图/1000文本。图像batch=64，沿用原文本编码批次和排序；cuda:0顺序评测，normalize(encode_image)与normalize(encode_text)内积，不使用mask、融合或rerank。四个原始JSON位于 [A2](formal500/A2) 与 [A3](formal500/A3)，评测器SHA256包含在 [运行前核验](formal500/evidence/preflight.json)。')
add()
checkpoints=[]
for arm in ('A2','A3'):
    a=r['audits'][arm]
    checkpoints.append([arm,'step500训练检查点',a['checkpoints'][-1]['sha256']])
    checkpoints.append([arm,'step500裸学生',a['export_verification']['bare_sha256']])
table(['实验','产物','SHA256'],checkpoints)
add('其他保存点的路径、字节数和SHA256见机器可读结果 `audits.*.checkpoints`。大checkpoint保留于服务器 `/root/lk_projects/SAID-nest-clip-v1/formal/{A2,A3}`；GitHub仅发布小体积报告、日志和证据。')
add()
add('效果与机制分开解读。A3总loss包含额外正则，不能用总loss大小直接比较模型效果；下面同时列出 `common_loss = loss − inc_weight × inc`。所有机制数值来自既有训练日志，最后50步指451–500的逐步算术均值。')
add('hard包含违例是O/E相对F的逐特征维度违例比例的平均，并非“有违例的样本比例”；全开/全关列才是对应样本比例。O/E IoU使用hard mask交集/并集。')
add()
for arm in ('A2','A3'):
    add(f'**{arm}：对齐与共同目标**（两个方向相加是单视图对齐loss，表中保留每个方向）')
    add()
    table(['step','总loss','common_loss','F I2T','F T2I','O I2T','O T2I','E I2T','E T2I'],[
        [s]+[f'{m[k]:.6f}' for k in ('loss','common_loss','F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i')]
        for s,m in sequence(arm)])
    add(f'**{arm}：mask选择**；单元格依次为 keep ratio / 全开比例 / 全关比例，均为0–1。')
    add()
    table(['step','F keep / open / closed','O keep / open / closed','E keep / open / closed'],[
        [s]+[' / '.join(f'{m[f"{v}_{k}"]:.6f}' for k in ('keep_ratio','all_open','all_closed')) for v in ('F','O','E')]
        for s,m in sequence(arm)])
    add(f'**{arm}：包含与重叠**；inc未加权，hard违例与IoU为0–1。')
    add()
    table(['step','inc','权重','hard包含违例','O/E IoU','有效局部数'],[
        [s]+[f'{m[k]:.6f}' for k in ('inc','inc_weight','hard_inclusion_violation','oe_iou')]+[f"{m['valid_global']:.2f}"]
        for s,m in sequence(arm)])

add('数据与运行成本：')
add()
table(['实验','观测样本数','局部有效比例','全局V范围','F-only更新数','回退原因计数','重复ID出现步数'],[
    [arm,a['samples_seen'],f"{100*a['valid_local_ratio']:.6f}%",f"{a['global_valid_min']}–{a['global_valid_max']}",
     a['f_only_steps'],json.dumps({k:v for k,v in a['view_reason_counts'].items() if k!='valid'},ensure_ascii=False),len(a['duplicate_steps'])]
    for arm,a in r['audits'].items()])
add('观测样本数按500次更新的图像样本行计数；有效比例仅代表本轮训练流，不冒充全训练集统计。重复ID按原一正例口径记录，不改变损失。完整逐步原因和重复ID见各组 `steps.jsonl`。')
add()
timings=[]
for arm,a in r['audits'].items():
    for identity,rank in zip(a['config']['ranks'],a['acceptance']['ranks']):
        timings.append([arm,rank['rank'],identity['pid'],identity['uuid'],f"{rank['seconds']:.3f}",
                        f"{rank['peak_allocated_gib']:.3f}",f"{rank['peak_reserved_gib']:.3f}"])
table(['实验','rank/GPU','PID','GPU UUID','循环耗时秒','峰值allocated GiB','峰值reserved GiB'],timings)
add('循环耗时包括DataLoader启动、同步更新、检查点写入及末尾一致性核验；完整进程耗时另包含模型构造等启动开销。')
add()
table(['命令阶段','退出码','进程耗时秒'],[[name,e['exit_code'],f"{e['wall_seconds']:.3f}"] for name,e in r['executions'].items()])
add('每组启动均通过四rank NCCL all_reduce=10与可导gather梯度=10测试；设备可见性/MIG/NCCL通信开关未修改。')
add()
add('四个诊断问题的定量依据（最后50步）：')
add()
table(['量','A2','A3','A3−A2'],[
    [key,f"{r['mechanism']['A2']['last50_mean'][key]:.6f}",f"{r['mechanism']['A3']['last50_mean'][key]:.6f}",
     f"{r['mechanism']['A3']['last50_mean'][key]-r['mechanism']['A2']['last50_mean'][key]:+.6f}"]
    for key in ('common_loss','inc','hard_inclusion_violation','F_keep_ratio','F_all_open','O_keep_ratio','E_keep_ratio','oe_iou')])
add('对四个诊断问题的回答：')
add()
r1=[x for x in delta if x['metric']=='R@1']
add('1. **step500原生检索**：'+ '；'.join(f"{x['dataset']} {x['direction']} R@1：{x['A2']:.3f}%→{x['A3']:.3f}%（{x['delta_pp']:+.3f} pp）" for x in r1)+f'。全部R@1/5/10合计{up}项上升、{down}项下降，结论仅限本次固定预算对照。')
a=r['mechanism']['A2']['last50_mean']; b=r['mechanism']['A3']['last50_mean']
add(f"2. **包含违例**：最后50步hard违例从{100*a['hard_inclusion_violation']:.3f}%到{100*b['hard_inclusion_violation']:.3f}%（{100*(b['hard_inclusion_violation']-a['hard_inclusion_violation']):+.3f} pp）；未加权软包含loss从{a['inc']:.6f}到{b['inc']:.6f}。两种指标均降低。")
add(f"3. **F是否趋向全开**：F keep ratio从{100*a['F_keep_ratio']:.3f}%升至{100*b['F_keep_ratio']:.3f}%（{100*(b['F_keep_ratio']-a['F_keep_ratio']):+.3f} pp），A3的F选择明显更宽、更接近全开。最后50步全开样本比例两组均为{100*b['F_all_open']:.3f}%，不能把平均91%左右的保留率写成每个mask完全全开。违例降低伴随这一变化，不能仅凭违例指标归因为语义选择质量改善。")
add(f"4. **O/E是否仍高度相似**：IoU均值从{a['oe_iou']:.6f}升至{b['oe_iou']:.6f}，仍有较高重叠；本轮A3没有呈现更强的O/E分离。该结论限于现有训练batch中的hard-mask选择日志。")
add()
add('限制：单次seed=0、500步；未做统计显著性检验。训练日志的IoU和包含违例不等于独立语义质量评价；F平均keep ratio高也不等于所有样本全开，应联合全开比例判断。沿用训练资产的SAM图像来源限制仍存在，未与Meta原图逐字节核对。没有选择中间checkpoint，也没有将历史S0/Clean/Full混作本轮同预算对照。')
add()
add('执行命令沿用 [run.sh](run.sh)，按 train500 A2→A3、export500 A2→A3、分别verify-export、coco A2→urban A2→coco A3→urban A3 的顺序。所有实际命令、开始/结束时间和退出码保存在 `formal500/evidence/*.execution.json`。报告完成后停止，不继续到1000步或3epoch，不启动额外消融。')
with (DEST/'FORMAL500_REPORT.md').open('x') as handle:
    handle.write('\n'.join(lines)+'\n')
