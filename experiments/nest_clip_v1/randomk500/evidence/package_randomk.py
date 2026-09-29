"""Publish small results and lossless compressed logs; no checkpoint/data copying."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

SOURCE=Path('/root/lk_projects/SAID-nest-clip-v1/randomk500')
DEST=Path('/root/lk_projects/SAID/experiments/nest_clip_v1/randomk500')
manifest=[]


def copy(source, relative, compress=False):
    destination=DEST/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    content=source.read_bytes()
    if compress:
        encoded=gzip.compress(content,mtime=0)
        destination.write_bytes(encoded)
        assert gzip.decompress(encoded)==content
    else:
        shutil.copyfile(source,destination)
    manifest.append(dict(path=relative,server_source=str(source),
                         source_sha256=hashlib.sha256(content).hexdigest(),source_bytes=len(content),
                         published_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                         published_bytes=destination.stat().st_size,compression='gzip' if compress else None))


for p in sorted((SOURCE/'evidence').rglob('*')):
    if p.is_file() and p.suffix in ('.json','.txt','.py','.exitcode'):
        copy(p,'evidence/'+str(p.relative_to(SOURCE/'evidence')))
for label,relative in [('smoke','smoke/A3-RandomK'),('formal','A3-RandomK')]:
    for name in ('config.json','acceptance.json','text_examples.json'):
        copy(SOURCE/relative/name,label+'/'+name)
    copy(SOURCE/relative/'steps.jsonl',label+'/steps.jsonl.gz',compress=True)
for name in ('coco_native.json','urban_native.json','export-check.json'):
    copy(SOURCE/'A3-RandomK'/name,'formal/'+name)
for name in ('REPORT.md','results.json'):copy(SOURCE/name,name)
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(DEST/'.gitattributes').write_text('# Preserve raw diagnostic output.\nevidence/*.txt -whitespace\n')
print(json.dumps(dict(files=len(manifest),published_bytes=sum(x['published_bytes'] for x in manifest)),indent=2))
