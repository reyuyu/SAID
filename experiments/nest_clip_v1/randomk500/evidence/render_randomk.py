"""Render the final RandomK comparison from measured results only."""
import json
from pathlib import Path

ROOT=Path('/root/lk_projects/SAID-nest-clip-v1/randomk500')
r=json.loads((ROOT/'results.json').read_text());a=r['formal_audit']
lines=[]
def add(s=''):lines.append(s)
def table(headers,rows):
    add('| '+' | '.join(headers)+' |');add('|'+'|'.join(['---']*len(headers))+'|')
    for row in rows:add('| '+' | '.join(map(str,row))+' |')
    add()

add('A3-RandomK：可见全文内部随机粒度采样的500步实验')
add()
add('完成必要测试、5步真实四卡smoke，再从共同step0独立训练到500步，严格导出并完成COCO/Urban原生检索。未重训固定A3/A2或Clean/Full，未改变目标权重、模型或优化配置，未追加消融或调参。')
add()
add('**本次随机K改善了Urban原生检索，但COCO出现回落，属于数据集间的权衡。** 相对固定A3，Urban I2T/T2I R@1分别提升1.000/1.300 pp；COCO I2T/T2I R@1分别下降0.200/0.376 pp。不能概括为随机K全面优于固定首句切分。')
add()
if a['shutdown_warning']:
    add('**运行诊断：正式训练在第500步后的DataLoader析构关闭阶段，两个worker报Aborted，异常标记为ignored in __del__；主任务退出码仍为0。** 四rank完成500更新、末尾参数一致性、全部日志及六个检查点核验通过，之后严格导出与native一致性核验也通过。该记录作为关闭阶段异常保留，不宣称运行无诊断。根因未确定，没有修改配置或重跑训练；原始记录见 [训练控制台](evidence/train500.console.txt)。')
    add()
delta=[x['delta_pp']['FixedA3'] for x in r['comparison']]
up=sum(x>0 for x in delta);down=sum(x<0 for x in delta)
add(f'主比较是 **A3-RandomK@500 vs 固定K=1的A3@500**。12项Recall中，RandomK上升{up}项、下降{down}项、持平{12-up-down}项。仅说明本次seed=0、500更新下的结果，不宣称统计显著，也不能据此单独证明包含项有效。')
add()
add('实际训练代码提交：`'+r['code_commit']+'`；参考提交：`'+r['reference_commit']+'`。`model/nested_semantic_mask.py`、原CLIP/MaskNetwork、严格导出及评测器均未修改；仅数据切分、配置传递/epoch设置和日志有变更。代码SHA256见 [运行前核验](evidence/preflight.json) 和 [正式配置](formal/config.json)。复现命令见 [README](README.md) 与 [run.sh](run.sh)。')
add()
add('F先由原 `text_views` 构造，原函数AST与190b477完全一致；没有按随机K重打包F。有效F有n个句段时，以SHA256(`sampling_seed:epoch:original_sample_id`)的大端整数为局部种子，调用独立 `random.Random(seed).randrange(1,n)`。sampling_seed=0、sample_id=index+1000。P为前K段，R为后n−K段，R保留末段；均独立tokenize，未截断长度不超过248，遇到超限直接报错，不重抽、不截断。单句/首句超长继续原F-only回退。')
add()
add('模型内arm仍为A3，包含权重min(1,s/200)、最高1；F/P/R双向对齐与稀疏公式、零有效样本路径和可导全局候选均不变。接口/log中的O表示P（变长前缀），E表示R（剩余描述）；本报告不将它们继续统称固定概述/展开。')
add()
add('配置：单机4张A100 80GB，world_size=4，batch=256/rank，accumulation=1，seed=0；完整训练样本1,245,901，1217更新/epoch、epochs=3、完整H=3651。smoke停止点5、正式停止点500；主干LR=1e-6、weight_decay=0.01、warmup200，mask LR=1e-3、weight_decay=0，原AdamW/cosine/BF16编码器/FP32损失、非重入checkpoint和score_chunk=64。初始化SHA256：`'+a['config']['init_sha256']+'`。')
add()
add('验证结果：18项测试通过，覆盖旧fixed_first回归、强制K=1的loss/梯度等价、n=2/多句/回退/末句/token预算、溢出不重抽、全局RNG隔离、跨epoch与0/2个spawn worker一致性，以及按n统计的有限样本K分布。固定K=1的loss/梯度沿用原容差；数学评分模块没有变更。详见 [测试日志](evidence/unit-tests.txt)。')
add()
smoke=r['smoke_audit']
add(f"smoke四rank均完成5更新，参数相对rank0最大差0，5步loss为{', '.join(f'{x:.6f}' for x in smoke['losses'])}；K=1占有效样本{100*smoke['k1_ratio']:.3f}%。正式前5步的样本/F/P-R/K摘要与smoke重放一致，但正式初始化重新读取共同step0。")
add()
add('正式运行四rank均完成500同步更新，日志1–500连续，loss/梯度有限，末尾参数差0，H=3651、stop_updates=500。保存step0/100/200/300/400/500；step0模型和optimizer与共同初态一致，各rank初始Python/NumPy/Torch CPU/CUDA RNG与原固定A3一致。')
add()
add('对原固定A3的核验共覆盖500步×4rank：显式sample_ids与同seed的DistributedSampler一致；每rank的固定切分参考摘要均匹配原A3三视图摘要。该摘要使用实际tokens_f以及旧固定切分文本，结合未改动的F打包规则核对样本/F；局部P/R本来就改变，不要求新的三视图摘要与旧A3一致。所有日志K均按sample_id/epoch重放通过。CPU日志中的参考视图不参与模型前向。')
add()
add('原生检索百分比及百分点差：')
add()
table(['数据集','方向','指标','固定A3 (%)','RandomK (%)','RandomK−固定A3 (pp)'],[
    [x['dataset'],x['direction'],x['metric'],f"{x['percent']['FixedA3']:.3f}",f"{x['percent']['RandomK']:.3f}",f"{x['delta_pp']['FixedA3']:+.3f}"] for x in r['comparison']])
add('原始Recall为0–1，展示乘100，差值严格为100×(new−old)。COCO canonical为5000图/25000文本、similarity_chunk512；Urban为1000图/1000文本；图像batch64，原文本编码/排序/数值协议，cuda:0顺序运行。只使用normalize(encode_image)与normalize(encode_text)内积，无mask、融合或rerank，仅评正式step500。')
add()
add('Clean/Full同为500更新的横向参考（原有本机结果，未重训或重测）：')
add()
table(['数据集','方向','指标','Clean@500 (%)','Full@500 (%)','RandomK (%)'],[
    [x['dataset'],x['direction'],x['metric'],f"{x['percent']['clean']:.3f}",f"{x['percent']['full']:.3f}",f"{x['percent']['RandomK']:.3f}"] for x in r['comparison']])
add('本次对Clean和Full均补查实际训练/裸学生文件SHA256、step500元信息、严格导出记录、原评测退出码、评测器源码和数据注释/描述指纹，均通过（[核验](evidence/baselines/verification.json)）。它们仍是不同方法的既有500步参考，不能用来单独归因随机K或包含项；没有混入3651步结果。全部相对参考的差值及原始JSON见 [results.json](results.json)。')
add()
add('随机采样统计：')
add()
add(f"正式共处理{a['samples']:,}个样本行，有效局部样本{a['valid_samples']:,}（{100*a['valid_ratio']:.4f}%），其中K=1为{a['k1_samples']:,}（{100*a['k1_ratio']:.4f}%）。全局有效数范围{a['global_valid_min']}–{a['global_valid_max']}，重复ID出现步数{len(a['duplicate_steps'])}。回退计数：{json.dumps({k:v for k,v in a['fallback_reasons'].items() if k!='valid'},ensure_ascii=False)}。")
add()
table(['n','有效样本数','观测K=1比例 (%)','均匀规则期望 (%)','按K计数'],[
    [n,sum(c.values()),f"{100*c.get('1',0)/sum(c.values()):.3f}",f"{100/(int(n)-1):.3f}",', '.join(f'{k}:{v}' for k,v in sorted(c.items(),key=lambda x:int(x[0]))) ]
    for n,c in a['K_histogram_by_n'].items()])
add('均匀性是条件于n的Uniform{1,…,n−1}；不要求有限样本计数完全均衡，也不要求跨不同n汇总后的K均匀。n=2必为K=1。下表只统计有效样本；token长度包含SOT/EOT且未截断。')
add()
table(['分布','最小','P25','中位数','P75','P95','最大','均值'],[
    [k]+[f'{v[field]:.3f}' if field=='mean' else v[field] for field in ('min','p25','median','p75','p95','max','mean')]
    for k,v in a['length_distributions'].items()])
add('逐样本sample_id/n/K与独立摘要在训练日志中，完整直方图位于results.json；少量真实F/P/R文本及其n/K见 [样例](formal/text_examples.json)。')
add()
add('最后50步（451–500）逐步均值：原固定A3的O/E是首句/其余，新RandomK的O/E是变长P/R。共同目标为total_loss−inc_weight×inc。局部文本任务的难度随切分改变，因此共同训练loss也不能替代原生检索成绩。')
add()
keys=['loss','common_loss','F_i2t','F_t2i','F_align','O_i2t','O_t2i','O_align','E_i2t','E_t2i','E_align',
      'inc','inc_weight','hard_inclusion_violation','F_keep_ratio','O_keep_ratio','E_keep_ratio',
      'F_all_open','O_all_open','E_all_open','F_all_closed','O_all_closed','E_all_closed','oe_iou','valid_global']
table(['日志量（O=P、E=R）','固定A3','RandomK','差值（原单位）'],[
    [k,f"{r['mechanism']['FixedA3']['last50'][k]:.6f}",f"{r['mechanism']['RandomK']['last50'][k]:.6f}",
     f"{r['mechanism']['RandomK']['last50'][k]-r['mechanism']['FixedA3']['last50'][k]:+.6f}"] for k in keys])
add('hard包含违例是逐特征维度违例比例的均值，全开/全关是样本比例。P/R IoU描述hard mask重叠，不直接证明语义互补。')
add()
old=r['mechanism']['FixedA3']['last50'];new=r['mechanism']['RandomK']['last50']
add(f"机制上，随机K的未加权包含loss从{old['inc']:.6f}变为{new['inc']:.6f}，hard包含违例从{100*old['hard_inclusion_violation']:.3f}%变为{100*new['hard_inclusion_violation']:.3f}%；F keep ratio从{100*old['F_keep_ratio']:.3f}%变为{100*new['F_keep_ratio']:.3f}%。P/R IoU从{old['oe_iou']:.6f}升至{new['oe_iou']:.6f}。本次没有观察到更低的包含违例或更低的局部mask重叠，不能把Urban收益归因为这两个指标的改善。")
add()
table(['rank/GPU','PID','GPU UUID','循环耗时秒','峰值allocated GiB','峰值reserved GiB'],[
    [h['rank'],identity['pid'],identity['uuid'],f"{h['seconds']:.3f}",f"{h['peak_allocated_gib']:.3f}",f"{h['peak_reserved_gib']:.3f}"]
    for identity,h in zip(a['config']['ranks'],a['acceptance']['ranks'])])
add('循环耗时含worker启动、更新、保存和末尾一致性检查；完整进程耗时包括模型初始化等额外开销。')
add()
table(['阶段','退出码','进程耗时秒'],[[name,x['exit_code'],f"{x['wall_seconds']:.3f}"] for name,x in r['executions'].items()])
add('启动时四rank NCCL all_reduce与可导gather梯度检查通过。没有降低batch、缩候选池、改变通信开关或结束其他进程。严格导出后optimizer step=500，native图像/文本embedding与训练模块接口最大差均为0。')
add()
table(['产物','SHA256'],[[f"step{x['step']}",x['sha256']] for x in a['checkpoints']]+[['step500裸学生',r['export_verification']['bare_sha256']]])
add('大文件保留于服务器 `/root/lk_projects/SAID-nest-clip-v1/randomk500/A3-RandomK`，GitHub只包含代码、配置、小体积报告/证据和压缩后的逐步日志。日志压缩为无损gzip，原始文件SHA256与压缩文件SHA256分别列在manifest中。')
add()
add('结论仅回答：在同一A3结构和训练目标下，本次随机K相对固定首句切分的表现如何。包含项权重没有变化；不能据此单独证明包含项有效，不对单seed小幅差值作显著性判断。保留原数据资产的来源限制，尤其SAM图片未与Meta原图逐字节核对。本轮到step500、评测与报告完成后停止，不继续到1000步或3epoch。')
(ROOT/'REPORT.md').write_text('\n'.join(lines).rstrip()+'\n')
