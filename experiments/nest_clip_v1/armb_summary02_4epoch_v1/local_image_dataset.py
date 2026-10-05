"""IO-only local mirror admission and evidence; native dataset values unchanged."""

import json
import os
from pathlib import Path

from train.nested_semantic_data import NestedDataset
from recovery.local_ssd_stage import IMAGES


class LocalImageDataset(NestedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.image_root.resolve() != IMAGES.resolve():
            raise RuntimeError("Training must read the verified local image mirror")
        self._path_proof_count = 0

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if self._path_proof_count < 8:
            record = json.loads(self._records[self._offsets[index]:self._offsets[index + 1]])
            path = (self.image_root / record["image"]).resolve()
            if not path.is_relative_to(IMAGES.resolve()):
                raise RuntimeError("Image path escaped local mirror")
            rank = int(os.environ["RANK"])
            evidence = Path(os.environ["SAID_S02_PHASE_LOCAL"]) / f"image-paths-rank{rank}-pid{os.getpid()}.jsonl"
            with evidence.open("a") as handle:
                handle.write(json.dumps(dict(rank=rank, pid=os.getpid(), sample_id=result["sample_id"],
                    resolved_image_path=str(path), basename=path.name, local_root=str(IMAGES),
                    native_getitem_used=True, path_from_NFS=False, dataset_return_unchanged=True)) + "\n")
            self._path_proof_count += 1
        return result
