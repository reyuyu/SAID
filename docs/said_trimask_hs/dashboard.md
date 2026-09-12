# 训练面板与后台 runner（S0-TriMask-HS v0.2）

## 1. 启动服务

```bash
cd /root/SAID-s0-trimask-hs-v02
setsid nohup /root/miniconda3/envs/said-smartclip/bin/python \
  tools/serve_training_dashboard.py \
  --run s0_trimask_hs_v02=runs_salu/said_s0_trimask_hs_v02/step500 \
  --run s0_trimask_v01=/root/SAID-s0-trimask-v01/runs_salu/said_s0_trimask_v01/step500 \
  > /root/trimask_hs_dashboard.log 2>&1 < /dev/null &
```

启动时会把实际的监听地址、PID、登记的 run 列表打印成一行 JSON（也在 `--help` 里说明）。默认绑定
`127.0.0.1:8765`；端口被占用时会依次尝试后续端口，并在 `port_fallback: true` 与日志里明确报告实际端口，
不会静默换端口。

## 2. 从你自己的电脑访问

远端服务只监听 loopback，所以需要一条 SSH 端口转发。命令模板（把 `<你的SSH目标>` 换成你实际使用的
别名或 `用户@主机`；我没有猜你的账号，也没有替你执行这条命令）：

```bash
ssh -N -L 8765:127.0.0.1:8765 <你的SSH目标>
```

然后打开 <http://127.0.0.1:8765>。若服务实际绑定的是备用端口（例如 8766），把命令和地址里的 8765 一起换掉。

**注意**：远端执行了 `ssh -L` 之后你的电脑才能访问；在没有做端口转发之前，页面是打不开的。

## 3. 为什么不用仓库里已有的 Streamlit 面板

仓库里确实有一个轻量只读面板 `tools/said_dashboard`（Streamlit + altair，只读、loopback、SSH 隧道，
理念与本面板一致）。它没有复用的原因是它面向的是一套**预先导出的注意力产物**（`outputs/salu_dashboard/`
下的 JSON/NPY/PNG，需要先跑一个导出步骤），而本面板要读的是**正在训练中的 run 的 JSONL 增量**，并且本轮
任务把 HTTP 端点和前端文件结构都写得很具体（§9.2/§9.3 要求可测的健康端点、游标式增量读取、路径穿越拒绝
等）。Streamlit 没有这些端点，那些测试就无从写起。

为此数据层被刻意做成与传输无关的 `tools/dashboard_data.py`：如果之后想做成 Streamlit 的一个页面，
直接 import 它即可，不需要改后端。

## 4. 只读端点

| 端点 | 说明 |
|---|---|
| `GET /health` | 健康检查，返回登记的 run 列表；同时声明 `read_only: true`、`gpu_used: false`、`checkpoints_loaded: false` |
| `GET /api/runs` | run 列表（含 `demo` 标记） |
| `GET /api/run/<id>/status` | 概览：phase、进度、ETA（标注“估计”）、耗时、显存、吞吐、gate 模式、λ |
| `GET /api/run/<id>/metrics?after=<cursor>` | JSONL 标量历史，游标为记录序号，只读新增字节 |
| `GET /api/run/<id>/masks` | 最新 mask 快照（512 维逐坐标均值等）与最近一次重统计标量 |
| `GET /api/run/<id>/evaluation` | COCO / Urban-1k 指标、基线与晋级门判定 |
| `GET /api/run/<id>/logs?limit=N` | 日志尾部、warning/error、脱敏后的实际命令 |
| `GET /api/run/<id>/metrics.csv` | 小型指标 CSV 下载（不含 caption、不含张量） |

页面用 `textContent` 写入所有动态文本；没有任何写入、执行命令、启动训练、kill 进程或上传脚本的接口
（POST 直接返回 501）。

## 5. 安全与解耦

* **run_id 只能来自启动时登记的注册表**，客户端无法提供路径，因此路径穿越不可表达；静态文件另外做了
  realpath containment 检查，只允许 `.html/.js/.css`。
* **不 import torch、不加载 checkpoint、不做前向、不占 GPU**：这一条由测试在独立解释器里断言
  （`"torch" in sys.modules` 必须为假）。页面刷新、关闭、断网或服务被杀都不会影响训练。
* **只读白名单文件名**：`run_status.json`、`config.json`、`run_summary.json`、`salu_log.jsonl`、
  `mask_snapshot.json`、`evaluation/*_canonical.json`、`evaluation/*_urban1k.json`。
* JSONL 增量读取：缓存字节偏移、忽略尚未写完的末行（下次轮询再读）、损坏的完整行记 warning 后跳过、
  文件被截断或轮换时只重置自己那条流；缺失值显示“暂无”，不伪造 0，也不把 10 步一次的标量插值成每步。
* 演示数据必须带 `demo: true`（页面标 `[DEMO]`），不会混进真实 run 列表。

## 6. 后台 runner

```bash
cd /root/SAID-s0-trimask-hs-v02
setsid nohup /root/miniconda3/envs/said-smartclip/bin/python \
  tools/trimask_hs_runner.py \
  --run-id s0_trimask_hs_v02 \
  --run-dir /root/SAID-s0-trimask-hs-v02/runs_salu/said_s0_trimask_hs_v02/step500 \
  --steps 500 --lambda-sparse-t 0.2 \
  > /root/trimask_hs_runner.out 2>&1 < /dev/null &
```

* 原子锁（`O_CREAT|O_EXCL` + PID）：第二个实例会被拒绝；PID 已消失的陈旧锁默认也拒绝启动，只有显式
  `--clear-stale-lock` 才会替换它。
* 启动前重新检查 GPU：`nvidia-smi -L` 必须成功并且能看到 4 张卡，`--query-compute-apps` 必须为空；
  `nvidia-smi` 本身失败一律当作“不可用”而不是空闲。
* 顺序执行 `train → 校验精确 step500 checkpoint → 导出裸学生 → COCO canonical → Urban-1k`，每步保留真实
  退出码；训练失败就直接停止，绝不去评估一个不存在或不完整的 checkpoint。
* 每个阶段原子写 `run_status.json`（临时文件 + `os.replace`），字段包含 `phase`、`started_at`、
  `updated_at`、`completed_steps`、各阶段退出码、`conclusion`、脱敏后的实际命令。前端只读它，从不写它。
* 面板挂掉不影响训练；训练失败时面板显示“失败”，不会因为进程退出就显示完成。

## 7. 已完成的验证

`tests/test_training_dashboard.py`（22 个用例）覆盖：健康端点、空 run 目录、半行 JSONL（先忽略、写完后
读到）、历史读取与增量游标、损坏完整行与超长行、日志截断/轮换重置、阶段与 ETA 语义（500/500 但评估未结束
时仍显示评估阶段；样本不足时不显示 ETA）、失败状态、mask 快照与交集字段、评估文件出现前后、路径穿越拒绝
（`/api/run/..%2f..`、`/static/../../etc/passwd` 等）、无写入方法、独立解释器里的“不 import torch / 不初始化
CUDA”断言、以及“只打开白名单文件”。

浏览器视觉验收：本轮**未执行**（没有可用的浏览器工具），页面渲染属于 NOT RUN，只完成了 HTTP/API 层验证
与静态资源返回检查。
