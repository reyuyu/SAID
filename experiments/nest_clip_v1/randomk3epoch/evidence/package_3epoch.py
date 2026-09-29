"""Package small evidence and derived compact trajectory, keeping full raw logs on server."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

SOURCE=Path('/root/lk_projects/SAID-nest-clip-v1/randomk3epoch')
DEST=Path('/root/lk_projects/SAID/experiments/nest_clip_v1/randomk3epoch')
manifest=[]
def copy(src,relative,compress=False):
    dst=DEST/relative;dst.parent.mkdir(parents=True,exist_ok=True)
    assert not dst.exists(),dst
    data=src.read_bytes()
    dst.write_bytes(gzip.compress(data,mtime=0) if compress else data)
    manifest.append(dict(path=relative,server_source=str(src),source_bytes=len(data),
                        source_sha256=hashlib.sha256(data).hexdigest(),
                        published_bytes=dst.stat().st_size,published_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
                        compression='gzip' if compress else None))
for p in sorted((SOURCE/'evidence').rglob('*')):
    if p.is_file() and p.suffix in ('.json','.txt','.py','.exitcode'):
        copy(p,'evidence/'+str(p.relative_to(SOURCE/'evidence')))
for p in sorted((SOURCE/'evaluation').rglob('*.json')):
    copy(p,'evaluation/'+str(p.relative_to(SOURCE/'evaluation')))
for name in ('config.json','acceptance.json','export-check.json'):
    copy(SOURCE/'A3-RandomK'/name,'training/'+name)
for name in ('results.json','REPORT.md'):copy(SOURCE/name,name)
copy(SOURCE/'trajectory_summary.jsonl','training/trajectory_summary.jsonl.gz',True)
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(DEST/'.gitattributes').write_text('# Preserve raw diagnostic logs.\nevidence/*.txt -whitespace\n')
print(json.dumps(dict(files=len(manifest),published_bytes=sum(x['published_bytes'] for x in manifest)),indent=2))
