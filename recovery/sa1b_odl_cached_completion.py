"""Complete an exclusively owned fully cached ODL task with zero network payload."""

import argparse
import os
from pathlib import Path
import signal

from openxlab.dataset.io.downloader import BigFileDownloader

import sa1b_recovery as recovery
from sa1b_opendatalab_probe import recover_complete_cache


def complete(filename, directory, old_pid, local_assembly=False):
    row = next(row for row in recovery.load(recovery.STATE)["shards"] if row["filename"] == filename)
    directory = Path(directory)
    target = recovery.ASSETS / "downloads/sa1b-opendatalab" / ("split-"+filename.removesuffix(".tar")) / "OpenDataLab___SA-1B"
    result = dict(started_utc=recovery.now(), completed=False, filename=filename, old_download_worker_pid=old_pid,
        mechanism="Official SDK native completion of fully cached ranges; old worker paused; no HTTP payload, no HF interruption",
        repository="OpenDataLab/SA-1B", dataset_id=6248, network_payload_bytes=0, cache_preserved=True)
    try:
        status = Path(f"/proc/{old_pid}/status").read_text()
        if "State:\tT" not in status:
            raise RuntimeError("Prior SDK worker must be paused before same-object local cache completion")
        downloader = BigFileDownloader(url="", filename="raw/"+filename, idx=0, download_dir=str(target),
                                       file_size=row["expected_size_bytes"])
        alternate = (Path("/root/.cache/said-recovery/odl-completed") / (filename.removesuffix(".tar")+"-"+str(os.getpid()))
                     if local_assembly else target / "raw" / (".sdk-completed-"+str(os.getpid())))
        alternate.mkdir(parents=True, exist_ok=False)
        if local_assembly:
            downloader.download_dir = str(alternate)
            downloader.prefix = ""
        else:
            downloader.prefix = str(alternate.relative_to(target))
        proof = recover_complete_cache(downloader, row)
        if proof is None:
            downloader._BigFileDownloader__done.set()
            raise RuntimeError("CACHE_ONLY completion requires exact full contiguous coverage; no network fallback")
        source = Path(proof["archive"])
        canonical = target / "raw" / filename
        if canonical.exists():
            recovery.verify_archive(canonical, row)
        else:
            if local_assembly:
                canonical.symlink_to(source)
            else:
                os.link(source, canonical)
        proof.update(canonical_archive=str(canonical), staged_native_assembly=str(source),
                     native_assembly_destination_reason="NFS EIO during bulk assembly; native SDK assembles in independent directory, pinned size/MD5/SHA verify, then no-overwrite canonical link",
                     assembly_backend="Persistent local SSD cache + project symlink" if local_assembly else "NAS alternate directory + hardlink")
        result.update(completed=True, finished_utc=recovery.now(), completed_cache_proof=proof)
        recovery.atomic_json(directory / "completed-cache-resume.json", proof)
        recovery.atomic_json(directory / "worker-result.json", result)
        os.kill(old_pid, signal.SIGKILL)
        print(filename, "SDK_NATIVE_COMPLETED_CACHE_REPAIR", flush=True)
    except BaseException as error:
        result.update(error_type=type(error).__name__, finished_utc=recovery.now())
        recovery.atomic_json(directory / "cache-repair-failed.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--old-pid", type=int, required=True)
    parser.add_argument("--local-assembly", action="store_true")
    args = parser.parse_args()
    complete(args.filename, args.directory, args.old_pid, args.local_assembly)
