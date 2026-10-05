"""Read-only resource evidence and bounded, authorized40-update replay."""

import argparse
import ctypes
import datetime
import errno
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/nest_clip_v1/armb_summary02_4epoch_v1"
RUNTIME = ROOT / "runtime/SAID-nest-clip-v1"
STEP0_SHA = "54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6"
GIB = 1 << 30


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    try:
        return Path(path).read_text()
    except (OSError, ValueError):
        return None


def pairs(path):
    text = read(path)
    return {key: int(value) for key, value in (line.split() for line in text.splitlines())} if text else {}


def pressure(path):
    text = read(path)
    result = {}
    for line in (text or "").splitlines():
        kind, *fields = line.split()
        result[kind] = {key: int(value) if key == "total" else float(value)
                        for key, value in (field.split("=") for field in fields)}
    return result


def memory_root():
    root = Path("/sys/fs/cgroup")
    return (root, 2) if (root / "memory.current").exists() else (root / "memory", 1)


def system_snapshot():
    root, version = memory_root()
    stats = pairs(root / "memory.stat")
    usage = read(root / ("memory.current" if version == 2 else "memory.usage_in_bytes"))
    limit = read(root / ("memory.max" if version == 2 else "memory.limit_in_bytes"))
    events = pairs(root / "memory.events") if version == 2 else pairs(root / "memory.oom_control")
    if version == 1:
        events["limit_hits_cumulative"] = int(read(root / "memory.failcnt") or 0)
    cpu = Path("/sys/fs/cgroup/cpu")
    return dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                cgroup_version=version, cgroup_path=str(root),
                memory_current=int(usage) if usage else None, memory_max=limit.strip() if limit else None,
                memory_stat=stats, memory_events=events,
                file=stats.get("file", stats.get("cache", 0)), inactive_file=stats.get("inactive_file", 0),
                active_file=stats.get("active_file", 0), anon=stats.get("anon", stats.get("rss", 0)),
                shmem=stats.get("shmem", 0), cpu_stat=pairs(cpu / "cpu.stat"),
                memory_PSI=pressure("/proc/pressure/memory"), io_PSI=pressure("/proc/pressure/io"),
                PSI_scope="host /proc/pressure; not cgroup-specific on this v1 host",
                cgroup_memory_PSI=pressure(root / "memory.pressure") if version == 2 else None,
                cgroup_io_PSI=pressure(root / "io.pressure") if version == 2 else None)


def bounded_reclaim(before):
    root, version = memory_root()
    path = root / "memory.reclaim"
    report = dict(attempted=False, successful=False, before=before, target_bytes=0,
                  global_drop_caches=False, image_files_modified=False)
    if version != 2 or not path.exists() or not os.access(path, os.W_OK):
        report["reason"] = "Current cgroup has no writable memory.reclaim; no system-level cache operation"
    elif before["inactive_file"] < 80 * GIB:
        report["reason"] = "Insufficient confirmed inactive file cache for a bounded64GiB reclaim"
    else:
        report.update(attempted=True, target_bytes=64 * GIB)
        try:
            path.write_text(str(64 * GIB))
        except OSError as error:
            report["kernel_error"] = dict(errno=error.errno, message=str(error))
            if error.errno != errno.EAGAIN:
                report["reason"] = "Reclaim unsupported/denied; no escalation or global fallback"
        time.sleep(2)
    report["after"] = system_snapshot()
    report["observed_usage_reduction_bytes"] = before["memory_current"] - report["after"]["memory_current"]
    report["successful"] = report["attempted"] and report["observed_usage_reduction_bytes"] >= 32 * GIB
    return report


class Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class Nvml:
    def __init__(self):
        self.error = None
        self.handles = []
        try:
            self.library = ctypes.CDLL("libnvidia-ml.so.1")
            self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            self.library.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(Utilization)]
            if self.library.nvmlInit_v2() != 0:
                raise RuntimeError("nvmlInit_v2 failed")
            for index in range(4):
                handle = ctypes.c_void_p()
                if self.library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                    raise RuntimeError("NVML GPU handle unavailable")
                self.handles.append(handle)
        except (OSError, RuntimeError) as error:
            self.error = str(error)

    def sample(self):
        if self.error:
            return dict(error=self.error)
        values = []
        for index, handle in enumerate(self.handles):
            rates = Utilization()
            code = self.library.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(rates))
            values.append(dict(index=index, gpu_percent=rates.gpu if code == 0 else None,
                               memory_percent=rates.memory if code == 0 else None, returncode=code))
        return values


def process_audit():
    processes, residual, open_files = [], [], []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        command = read(entry / "comm")
        if not command:
            continue
        command = command.strip()
        status = read(entry / "status") or ""
        parent = next((int(line.split()[1]) for line in status.splitlines() if line.startswith("PPid:")), None)
        processes.append(dict(pid=int(entry.name), ppid=parent, comm=command))
        try:
            arguments = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            arguments = b""
        if command in ("pt_data_worker", "python", "python3", "python3.10"):
            if any(term in arguments for term in (b"final_manifest_audit", b"multiprocessing.spawn", b"--multiprocessing-fork", b"train_nested_semantic_mask")):
                residual.append(dict(pid=int(entry.name), ppid=parent, comm=command))
        try:
            for descriptor in (entry / "fd").iterdir():
                try:
                    target = os.readlink(descriptor)
                except OSError:
                    continue
                if str(ROOT / "local_assets/training/ShareGPT4V") in target or "required_training_images.jsonl" in target:
                    open_files.append(dict(pid=int(entry.name), fd=descriptor.name, path=target))
        except (OSError, PermissionError):
            pass
    return dict(processes=processes, residual_audit_or_training_workers=residual,
                open_training_or_audit_files=open_files, workers_all_exited=not residual,
                killed_processes=[], removed_images=0)


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values):
    return dict(count=len(values), median=statistics.median(values), p95=quantile(values, .95), max=max(values)) if values else None


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def run():
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.recovery_full import sha
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = RUNTIME / ("resource-stall-v2-" + stamp)
    local = Path("/tmp/said-resource-stall-v2") / stamp
    run_root.mkdir(parents=True, exist_ok=False)
    local.mkdir(parents=True, exist_ok=False)
    dump(local / "paths.json", dict(output_dir=str(run_root / "train"), runtime=str(run_root)))
    process = process_audit()
    if not process["workers_all_exited"]:
        raise RuntimeError("Residual workers found; preserve and investigate before replay")
    before = system_snapshot()
    reclaim = bounded_reclaim(before)
    disks = {name: subprocess.check_output(command, text=True) for name, command in
             (("lsblk", ["lsblk", "-o", "NAME,TYPE,SIZE,ROTA,FSTYPE,MOUNTPOINT"]), ("df", ["df", "-hT"]))}
    preflight = dict(process_audit=process, cache_reclaim=reclaim, disks=disks,
                     data_audit_repeated=False, data_moved=False, formal500_or_full_authorized=False,
                     native_source_sha256={name: sha(ROOT / name) for name in
                         json.loads((ROOT / "recovery/evidence/s02-smoke-command.json").read_text())["code_sha256"]},
                     frozen_config_sha256=sha(ROOT / "recovery/configs/summary02.json"),
                     local_telemetry=str(local), runtime=str(run_root), maximum_updates=40, horizon=4868)
    for name, digest in preflight["native_source_sha256"].items():
        assert digest == json.loads((ROOT / "recovery/evidence/s02-smoke-command.json").read_text())["code_sha256"][name]
    assert sha(RUNTIME / "shared/step000000.pt") == STEP0_SHA
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    dump(run_root / "preflight.json", preflight)
    dump(EXP / "RESOURCE_DIAGNOSTIC_PREFLIGHT_V2.json", preflight)
    policy_path = ROOT / "recovery/evidence/recovery-operation-policy.json"
    policy = json.loads(policy_path.read_text())
    policy.update(formal_training_authorized=False, resume_allowed=False, resource_diagnostic_authorized=True,
                  maximum_diagnostic_updates=40, resource_safety_gate="unchanged:3 consecutive full cycles>3seconds",
                  reason="User authorizes one fresh common-step0 diagnostic replay, never500/full")
    dump(policy_path, policy)
    command = [str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nnodes=1", "--nproc-per-node=4", "--max-restarts=0",
               "-m", "experiments.nest_clip_v1.armb_summary02_4epoch_v1.phase_train_v2", "--config", str(ROOT / "recovery/configs/summary02.json"),
               "--init-state", str(RUNTIME / "shared/step000000.pt"), "--index-dir", str(RUNTIME / "data_index"),
               "--image-root", str(ROOT / "local_assets/training/ShareGPT4V"), "--output-dir", str(run_root / "train"),
               "--run-type", "formal", "--max-updates", "40"]
    nvml = Nvml()
    environment = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
                       SAID_PHASE_LOCAL=str(local), SAID_PHASE_SUPERVISOR_PID=str(os.getpid()))
    dump(run_root / "command.json", dict(command=command, diagnostic_only=True, cwd=str(ROOT)))
    with (run_root / "replay.log").open("w") as output, (local / "system-monitor.jsonl").open("w") as monitor:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                monitor.write(json.dumps(dict(system=system_snapshot(), gpu=nvml.sample())) + "\n")
                monitor.flush()
                steps_path = run_root / "train/steps.jsonl"
                completed = len(rows(steps_path)) if steps_path.exists() else 0
                dump(EXP / "RESOURCE_DIAGNOSTIC_PROGRESS_V2.json", dict(pid=child.pid, completed_updates=completed,
                     status="DIAGNOSTIC_RUNNING", maximum_updates=40, horizon=4868, runtime=str(run_root)))
                time.sleep(2)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=30)
            raise
    dump(run_root / "exit.json", dict(returncode=child.returncode, finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    analyze(run_root, local, preflight, child.returncode)
    policy.update(resource_diagnostic_authorized=False, resource_diagnostic_completed=True,
                  formal_training_authorized=False, resume_allowed=False)
    dump(policy_path, policy)


def analyze(run_root, local, preflight, returncode):
    native = run_root / "train"
    steps = rows(native / "steps.jsonl")
    cycles = rows(native / "cycle_timing.jsonl")
    phases = {rank: rows(local / f"rank{rank}.jsonl") for rank in range(4)}
    phase_map = {rank: {row["step"]: row for row in records} for rank, records in phases.items()}
    merged = []
    for cycle in cycles:
        step = cycle["step"]
        records = [phase_map[rank][step] for rank in range(4)]
        merged.append(dict(step=step, full_cycle_s=cycle["four_rank_max_seconds"], ranks=records))
    acceptance = json.loads((native / "acceptance.json").read_text()) if (native / "acceptance.json").exists() else None
    streak, triggered = 0, False
    for cycle in cycles:
        if cycle["step"] > 1:
            streak = streak + 1 if cycle["four_rank_max_seconds"] > 3 else 0
            triggered = triggered or streak >= 3
    ready = (returncode == 0 and acceptance and acceptance["passed"] and len(steps) == 40 and not triggered and
             all(cycle["four_rank_max_seconds"] <= 3 for cycle in cycles if cycle["step"] >= 31))
    status = "RESOURCE_GATE_READY_FOR_500" if ready else "RESOURCE_GATE_NOT_READY_FOR_500"
    result = dict(status=status, completed_updates=len(steps), horizon=4868, returncode=returncode,
                  native_three_consecutive_gate_triggered=triggered, cycles_all=summary([row["four_rank_max_seconds"] for row in cycles]),
                  cycles_steps7_to40=summary([row["four_rank_max_seconds"] for row in cycles if row["step"] >= 7]),
                  cycles_steps31_to40=summary([row["four_rank_max_seconds"] for row in cycles if row["step"] >= 31]),
                  native_acceptance=acceptance, preflight=preflight, phase_timings=merged,
                  cache_reclaim_performed=preflight["cache_reclaim"]["attempted"], no_500_or_full_started=True,
                  phase_semantics="CUDA events are stream elapsed, not exclusive kernel time. Backward includes DDP; explicit collective wall time is a subset when nested, not additive.")
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.reproduction_train_gate import compare_prefix
    smoke = Path(json.loads((ROOT / "recovery/evidence/s02-smoke-command.json").read_text())["quarantined_output_dir"])
    result["prefix_stream_gate"] = compare_prefix(steps[:5], rows(smoke / "steps.jsonl"))
    slow = [record for record in merged if record["step"] >= 7 and record["full_cycle_s"] > 3]
    result["slow_steps"] = slow
    dump(run_root / "result.json", result)
    dump(EXP / "RESOURCE_STALL_DIAGNOSIS_V2.json", result)
    dump(EXP / "RESOURCE_DIAGNOSTIC_PROGRESS_V2.json", dict(status=status, completed_updates=len(steps), runtime=str(run_root), training_running=False))
    print(json.dumps(dict(status=status, completed_updates=len(steps), stats=result["cycles_steps7_to40"], triggered=triggered)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.parse_args()
    run()
