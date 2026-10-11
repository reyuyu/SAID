"""Create the isolated L14 HNS common step0 from the official OpenAI weights."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np, torch
from model import longclip
from model.backbone import L14_SHA256, L14_URL, validate_backbone
from model.balanced_hparam_search import BalancedSearch, hparams
from train.train_nested_semantic_mask import state_digest, build_optimizer

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=Path.home()/'.cache/clip/ViT-L-14.pt'
OUT=Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/hns-v1-l14-pure3epoch3651-v1/shared/step000000.pt')
EXP=ROOT/'experiments/nest_clip_v1/hns_v1_l14_pure3epoch3651_v1'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def seed_all(seed=0):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def main():
    assert sha(WEIGHTS)==L14_SHA256
    assert not OUT.exists(), OUT
    seed_all(0)
    clip,_=longclip.load_from_clip('ViT-L/14',device='cpu',args=argparse.Namespace())
    dims=validate_backbone(clip,'ViT-L/14')
    cfg=json.loads((ROOT/'configs/hns_v1_l14_pure3epoch3651_v1.json').read_text())
    module=BalancedSearch(clip.float(), arm=cfg['arm'], search_hparams=hparams(cfg),
                          inclusion_hierarchy='detail_chain', view_sparsity_weights=[1.,2.,2.],
                          hns_enabled=True, hns_beta=[2.,2.], fusion='balanced_stack', visual='patch',
                          checkpoint_encoders=False, image_chunk=128, text_chunk=128,
                          condition_mode='dual_branch', shuffle_seed=0, checkpoint_pair_blocks=False)
    opt=build_optimizer(module)
    assert not opt.state
    assert module.fusion_branch.visual_adapter.weight.shape==(768,1024)
    assert module.fusion_branch.gate.weight.shape==(768,1536)
    assert module.fusion_branch.visual_tokens==256
    model_sha=state_digest(clip.state_dict())
    adapter_sha=state_digest(module.fusion_branch.state_dict())
    payload=dict(model=clip.state_dict(), optimizer=opt.state_dict(), completed_steps=0,
                 rng_cpu=torch.get_rng_state(), provenance=dict(source='OpenAI CLIP + original random MaskNetwork',
                     base_model='ViT-L/14', seed=0, original_clip_sha256=L14_SHA256,
                     original_clip_url=L14_URL, constructor='longclip.load_from_clip; positional extension 77->248; unchanged MaskNetwork; no training',
                     frozen_position=True, trainable_residual_position=True,
                     architecture=dims, model_sha256=model_sha, adapter_initialization_sha256=adapter_sha))
    OUT.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,OUT)
    report=dict(passed=True, checkpoint=str(OUT), sha256=sha(OUT), official_weight_sha256=sha(WEIGHTS),
                model_sha256=model_sha, adapter_initialization_sha256=adapter_sha, dimensions=dims,
                visual_tokens=module.fusion_branch.visual_tokens, adapter_shape=list(module.fusion_branch.visual_adapter.weight.shape),
                gate_shape=list(module.fusion_branch.gate.weight.shape), optimizer_state_empty=True,
                hns_enabled=True, hns_beta=[2,2], view_sparsity_weights=[1,2,2], positional_extension=[77,248],
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    EXP.mkdir(parents=True,exist_ok=True); (EXP/'L14_INITIALIZATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
