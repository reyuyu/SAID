"""Create a separate OpenAI L14 step0; never load a B16 checkpoint into L14."""
import argparse
import json
from pathlib import Path

import torch

from model import longclip
from model.backbone import validate_backbone, L14_SHA256, L14_URL
from model.balanced_hparam_search import BalancedSearch
from train.nested_semantic_data import file_sha
from train.train_nested_semantic_mask import seed_all, atomic_save, state_digest
from experiments.nest_clip_v1.runtime_optimization_v1.probe import optimizer_for, RUN


def main():
    torch.set_num_threads(4);seed_all(0)
    config=json.loads((RUN/'configs/original-reference.json').read_text())
    config.update(base_model='ViT-L/14',remainder_mode='compact')
    pretrained=Path.home()/'.cache/clip/ViT-L-14.pt'
    assert file_sha(pretrained)==L14_SHA256
    clip,_=longclip.load_from_clip('ViT-L/14',device='cpu',args=argparse.Namespace())
    clip.float();dimensions=validate_backbone(clip,'ViT-L/14')
    source=torch.jit.load(str(pretrained),map_location='cpu').state_dict()
    checked=0
    for name,value in clip.state_dict().items():
        if name.startswith('mask_net.') or name in ('positional_embedding','positional_embedding_res'):
            continue
        torch.testing.assert_close(value,source[name].float(),atol=0,rtol=0,msg=name);checked+=1
    del source
    module=BalancedSearch(clip,search_hparams=config,fusion='balanced_stack',visual='patch',checkpoint_encoders=False)
    assert module.fusion_branch.visual_adapter.weight.shape==(768,1024)
    assert module.fusion_branch.gate.weight.shape==(768,1536)
    assert module.fusion_branch.gate.bias is None and module.fusion_branch.gate.weight.count_nonzero()==0
    assert module.fusion_branch.visual_tokens==256
    optimizer=optimizer_for(module,config);assert not optimizer.state
    provenance=dict(source='OpenAI CLIP + original random MaskNetwork',base_model='ViT-L/14',
        pretrained_url=L14_URL,pretrained_sha256=L14_SHA256,dimensions=dimensions,
        pretrained_tensors_checked_exactly=checked,seed=0,independent_from_B16=True,
        mask_network_sha256=state_digest(clip.mask_net.state_dict()),
        visual_adapter_shape=[768,1024],gate_shape=[768,1536],mask_dim=768)
    path=RUN/'shared/l14-step000000.pt';path.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(dict(model=clip.state_dict(),adapter=module.fusion_branch.state_dict(),
        optimizer=optimizer.state_dict(),completed_steps=0,config=config,provenance=provenance),path)
    evidence=dict(passed=True,initial_checkpoint=str(path),initial_checkpoint_sha256=file_sha(path),provenance=provenance)
    output=Path(__file__).resolve().parent/'evidence';output.mkdir(exist_ok=True)
    (output/'l14-initialization.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
