"""Windows download/upload pipeline for the pinned 51-shard SA-1B candidate.

SSH credentials come only from SAID_SSH_HOST/PORT/USER/PASSWORD environment
variables. Host identity must already exist in the SSH known_hosts file.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import threading
import time

REPO = 'hanlincs/InternVL-SA1B-Caption-WebDataset'
REVISION = '4cdaea026f51899bb88d24d423121d2106a943ba'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def connect():
    import paramiko
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(os.environ['SAID_SSH_HOST'], port=int(os.environ['SAID_SSH_PORT']),
                   username=os.environ['SAID_SSH_USER'], password=os.environ['SAID_SSH_PASSWORD'],
                   timeout=20)
    client.get_transport().set_keepalive(20)
    return client


def sftp_batch(local, destination, offset):
    for value in (local, destination):
        if any(char in value for char in ('"', '\n', '\r')):
            raise ValueError('unsupported SFTP batch path')
    # OpenSSH reput fails when the remote path does not exist.
    return '%s "%s" "%s"\n' % ('reput' if offset > 0 else 'put', local, destination)


def remote(client, command):
    _, out, err = client.exec_command(command)
    data = out.read().decode('utf-8', errors='replace')
    error = err.read().decode('utf-8', errors='replace')
    if out.channel.recv_exit_status():
        raise RuntimeError(data + error)
    return data.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--local-dir', type=Path, required=True)
    parser.add_argument('--remote-dir', required=True)
    parser.add_argument('--remote-repo', required=True)
    parser.add_argument('--remote-python', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--native-sftp', action='store_true', help='Use native sftp reput; SSH_ASKPASS must be configured')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=50)
    args = parser.parse_args()
    if not 0 <= args.start <= args.end <= 50:
        raise ValueError('only shards 000000..000050 are authorized')
    args.local_dir.mkdir(parents=True, exist_ok=True)
    metadata = {x['path']: x for x in json.loads(args.metadata.read_text(encoding='utf-8'))}
    max_size = max(metadata['data/sa_%06d.tar' % i]['size'] for i in range(args.start, args.end + 1))
    already_present = sum(p.stat().st_size for p in args.local_dir.glob('sa_*.tar'))
    if shutil.disk_usage(args.local_dir).free + already_present < max_size * args.workers + 2 * 1024**3:
        raise RuntimeError('insufficient disk reserve for bounded pipeline')
    stop = threading.Event()
    report_lock = threading.Lock()
    receipts_path = args.local_dir / 'transfer_receipts.json'
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}

    def save(shard, values):
        with report_lock:
            receipts[shard] = dict(receipts.get(shard, {}), **values)
            temporary = receipts_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(receipts, indent=2, ensure_ascii=False), encoding='utf-8')
            temporary.replace(receipts_path)
        print(json.dumps(dict(shard=shard, **values), ensure_ascii=False), flush=True)

    def process(index):
        if stop.is_set():
            return
        shard = '%06d' % index
        filename = 'sa_%s.tar' % shard
        info = metadata['data/' + filename]
        expected = info['lfs']['oid']
        path = args.local_dir / filename
        destination = args.remote_dir.rstrip('/') + '/' + filename
        client = None
        try:
            started = time.monotonic()
            client = connect()
            # Resume without redownloading already verified server files.
            remote_hash = remote(client, 'if test -f {0}; then sha256sum {0}; fi'.format(shlex.quote(destination)))
            already_verified = remote_hash.split()[0] == expected if remote_hash else False
            if not already_verified:
                client.close()
                client = None
                if not path.exists() or path.stat().st_size != info['size']:
                    url = 'https://hf-mirror.com/datasets/%s/resolve/%s/data/%s?download=true' % (REPO, REVISION, filename)
                    log = args.local_dir / (filename + '.download.log')
                    with log.open('w') as out:
                        subprocess.run(['curl.exe', '-sS', '-L', '--fail', '--retry', '3', '--connect-timeout', '30',
                                        '--speed-time', '60', '--speed-limit', '1024', '-C', '-', url, '-o', str(path)],
                                       stdout=out, stderr=out, check=True)
                local_hash = digest(path)
                if path.stat().st_size != info['size'] or local_hash != expected:
                    raise RuntimeError('local SHA256/size mismatch; STOP')
                save(shard, {'download_completed': True, 'local_sha256': local_hash,
                             'size': info['size'], 'download_seconds': round(time.monotonic() - started, 2)})
                if stop.is_set():
                    return
                client = connect()
                sftp = client.open_sftp()
                partial = destination + '.partial'
                try:
                    offset = sftp.stat(partial).st_size
                except FileNotFoundError:
                    offset = 0
                if offset > path.stat().st_size:
                    raise RuntimeError('remote partial larger than expected')
                if args.native_sftp:
                    batch = args.local_dir / (filename + '.sftp.batch')
                    batch.write_text(sftp_batch(path.resolve().as_posix(), partial, offset), encoding='utf-8')
                    with (args.local_dir / (filename + '.upload.log')).open('w') as log:
                        subprocess.run(['sftp', '-o', 'BatchMode=no', '-o', 'StrictHostKeyChecking=yes',
                                        '-B', '262144', '-R', '64', '-P', os.environ['SAID_SSH_PORT'],
                                        '-b', str(batch), os.environ['SAID_SSH_USER'] + '@' + os.environ['SAID_SSH_HOST']],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log, check=True)
                else:
                    with path.open('rb') as source, sftp.open(partial, 'ab', bufsize=1024 * 1024) as target:
                        source.seek(offset)
                        target.set_pipelined(True)
                        while True:
                            block = source.read(1024 * 1024)
                            if not block:
                                break
                            target.write(block)
                actual = remote(client, 'sha256sum ' + shlex.quote(partial)).split()[0]
                if actual != local_hash:
                    raise RuntimeError('local/remote SHA256 mismatch; STOP; local retained')
                sftp.posix_rename(partial, destination)
                sftp.close()
                save(shard, {'upload_completed': True, 'hash_verified': True, 'remote_sha256': actual})
                # User explicitly permits deletion only after matching local/remote SHA256.
                if path.resolve().parent != args.local_dir.resolve():
                    raise RuntimeError('local path escaped working directory')
                path.unlink()
                save(shard, {'local_removed_after_hash_match': True})
            else:
                save(shard, {'upload_completed': True, 'hash_verified': True, 'remote_sha256': expected,
                             'server_file_reused': True})
            if stop.is_set():
                return
            command = ('cd %s && %s -m tools.data.prepare_hf_sa1b ingest --root %s --shard %s' %
                       tuple(shlex.quote(v) for v in (args.remote_repo, args.remote_python, args.data_root, shard)))
            result = remote(client, command)
            save(shard, {'ingested': True, 'error': None, 'elapsed_seconds': round(time.monotonic() - started, 2),
                         'ingest_result': json.loads(result)})
        except Exception as exc:
            stop.set()
            save(shard, {'error': '%s: %s' % (type(exc).__name__, exc)})
            raise
        finally:
            if client is not None:
                client.close()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [executor.submit(process, i) for i in range(args.start, args.end + 1)]
        errors = []
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('Pipeline stopped: ' + '; '.join(errors))


if __name__ == '__main__':
    main()
