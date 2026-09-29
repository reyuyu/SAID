"""Copy small formal500 evidence byte-for-byte; never copy data or model weights."""
import hashlib
import json
from pathlib import Path
import shutil

SOURCE = Path('/root/lk_projects/SAID-nest-clip-v1')
REPO = Path('/root/lk_projects/SAID')
DEST = REPO / 'experiments/nest_clip_v1/formal500'
DEST.mkdir(exist_ok=False)
manifest = []


def copy(src, relative):
    destination = DEST / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert src.stat().st_size < 10*1024*1024, src
    content = src.read_bytes()
    shutil.copyfile(src, destination)
    digest = hashlib.sha256(content).hexdigest()
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
    manifest.append(dict(path=relative, sha256=digest, bytes=len(content), server_source=str(src)))


for src in sorted((SOURCE / 'formal500-evidence').iterdir()):
    if src.is_file() and src.suffix in ('.json', '.txt', '.py'):
        copy(src, 'evidence/' + src.name)
for arm in ('A2','A3'):
    for name in ('config.json','acceptance.json','steps.jsonl','coco_native.json','urban_native.json'):
        copy(SOURCE / f'formal/{arm}/{name}', f'{arm}/{name}')
(DEST/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
(DEST/'.gitattributes').write_text('# Raw console logs are preserved byte-for-byte.\nevidence/*.console.txt -whitespace\n')
print(json.dumps(dict(files=len(manifest), bytes=sum(x['bytes'] for x in manifest)), indent=2))
