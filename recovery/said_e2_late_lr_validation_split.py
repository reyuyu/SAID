"""Read-only leakage screen of the pre-existing skipped1000 candidate set."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from recovery import said_e2_late_lr_fourarm as c


def first_records(path, count):
    decoder=json.JSONDecoder();buffer='';pos=0;opened=False
    with Path(path).open() as stream:
        for _ in range(count):
            while True:
                while pos<len(buffer) and buffer[pos] in ' \r\n\t,':pos+=1
                if not opened and pos<len(buffer):
                    assert buffer[pos]=='[';pos+=1;opened=True
                try:
                    value,end=decoder.raw_decode(buffer,pos)
                except json.JSONDecodeError:
                    buffer=buffer[pos:]+stream.read(1<<20);pos=0
                    continue
                pos=end;yield value;break


def main():
    metadata=c.read(c.protocol.local.INDEX/'metadata.json')
    candidates=list(first_records(metadata['annotation'],1000))
    images={r['image'] for r in candidates}
    caption_hash=lambda text:hashlib.sha256(text.strip().encode()).hexdigest()
    captions={caption_hash(r['conversations'][1]['value']) for r in candidates}
    candidate_duplicate_images=1000-len(images)
    repeated_images=Counter();repeated_captions=0;checked=0
    with (c.protocol.local.INDEX/'records.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line);checked+=1
            if r['image'] in images:repeated_images[r['image']]+=1
            if caption_hash(r['caption']) in captions:repeated_captions+=1
    assert checked==1245901
    public_names=set()
    bench=c.PROJECT/'local_assets/retrieval_benchmarks/manifests'
    for name in ('flickr30k_test1k.jsonl','docci_test.jsonl','long_dci_reconstructed.jsonl'):
        with (bench/name).open() as stream:
            for line in stream:
                record=json.loads(line)
                for key,value in record.items():
                    if isinstance(value,str) and value.lower().endswith(('.jpg','.jpeg','.png')):
                        public_names.add(Path(value).name)
    coco=c.PROJECT/'local_assets/evaluation/coco/val2017'
    public_names.update(p.name for p in coco.glob('*.jpg'))
    urban=c.PROJECT/'local_assets/evaluation/Urban1k/Urban1k'
    public_names.update(p.name for p in urban.rglob('*.jpg'))
    collisions=sorted(image for image in images if Path(image).name in public_names)
    c.dump(c.EXP/'INDEPENDENT_VALIDATION_SCREEN.json',dict(status='UNVERIFIED_NOT_USED_FOR_SELECTION',
        candidate_definition='Original annotation positions0..999, excluded by native training index',
        candidate_count=1000,candidate_unique_image_paths=len(images),candidate_duplicate_image_paths=candidate_duplicate_images,
        training_records_scanned=checked,candidate_images_repeated_in_training=dict(repeated_images),
        candidate_exact_captions_repeated_in_training=repeated_captions,
        public_test_filename_collisions=collisions,
        byte_pixel_perceptual_duplicates='UNVERIFIED',public_test_semantic_overlap='UNVERIFIED',
        independent_retrieval_protocol='NOT_ESTABLISHED',models_evaluated_on_candidate=False,
        decision='No trusted independent set established; all predefined public results are exploratory',
        annotation_sha256=metadata['annotation_sha256'],training_index_sha256=metadata['records_sha256']))


if __name__=='__main__':main()
