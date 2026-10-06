"""Bounded read-only live continuation status; no Torch imports or raw output."""
import json
from pathlib import Path
import statistics

RUN=Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-local500-20261006')
PHASE=Path('/root/said_s02_stage500/formal-full-phase-20261006')


def tail(path,bytes_limit=65536):
    if not path.exists():
        return []
    with path.open('rb') as stream:
        stream.seek(0,2)
        size=stream.tell()
        offset=max(0,size-bytes_limit)
        stream.seek(offset)
        lines=stream.read().splitlines()
        if offset:
            lines=lines[1:]
    rows=[]
    for line in lines:
        try:
            rows.append(json.loads(line))
        except (ValueError,UnicodeError):
            pass
    return rows


def status():
    cycles=tail(RUN/'step4868/cycle_timing.jsonl')[-60:]
    latest=tail(RUN/'full-resource-telemetry.jsonl')
    resource=latest[-1]['system'] if latest else {}
    native=tail(RUN/'step4868/steps.jsonl',262144)
    result=dict(step=cycles[-1]['step'] if cycles else 500,
        recent_full_cycle_median_s=round(statistics.median(c['four_rank_max_seconds'] for c in cycles),4) if cycles else None,
        recent_full_cycle_max_s=round(max(c['four_rank_max_seconds'] for c in cycles),4) if cycles else None,
        recent_rank_data_wait_median_s={str(r):round(statistics.median(p['data_wait_s'] for p in samples),6)
            for r in range(4) if (samples:=tail(PHASE/f'rank{r}.jsonl'))},
        cgroup_GiB=round(resource.get('memory_current',0)/2**30,3),
        file_cache_GiB=round(resource.get('file',0)/2**30,3),oom_kill=resource.get('memory_events',{}).get('oom_kill'),
        command=latest[-1]['command'] if latest else None,
        last_loss=native[-1].get('loss') if native else None,
        IO_PSI_full_avg10=resource.get('io_PSI',{}).get('full',{}).get('avg10'),
        memory_PSI_full_avg10=resource.get('memory_PSI',{}).get('full',{}).get('avg10'),
        completed=(RUN/'full-supervisor-result.json').exists())
    if result['completed']:
        saved=json.loads((RUN/'full-supervisor-result.json').read_text())
        result.update(error=saved['error'],status=(saved['result'] or {}).get('status'))
    print(json.dumps(result))


if __name__=='__main__':
    status()
