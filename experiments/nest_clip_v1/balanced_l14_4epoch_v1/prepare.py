"""Prepare a distinct OpenAI L14 Balanced step0 and validate real native interfaces."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from model import longclip
from model.backbone import L14_URL, L14_SHA256, load_native_state, validate_backbone
from model.balanced_hparam_search import BalancedSearch,hparams
from train.train_nested_semantic_mask import atomic_save,build_optimizer,seed_all,state_digest
from train.nested_semantic_data import file_sha


REPO=Path(__file__).resolve().parents[3]
EXP=Path(__file__).resolve().parent
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/balanced_l14_4epoch_v1')
INIT=RUN/'shared/step000000.pt'
CONFIG=REPO/'configs/nest_balanced_l14_4epoch_v1.json'
REFERENCE_COMMIT='14653c92c6da9d552a2b624ab169eaaa275cdde8'
B16_SHA='f36438934947ed7f1523fe87e877a0e62dd807dc8a7c89c751eac537e0ab64bb'
B16_PATH=Path('/root/lk_projects/SAID-nest-clip-v1/three_followup_v1/four_epoch/trials/6d44ae8d5c34438e672287b09f3e6930af01473a4aa4905cd90c02c259b2a4a8/step4868/student_step4868.pt')


def expected_positions(original):
    source=original.half()
    length,width=source.shape
    target=source.new_zeros(4*length-60,width)
    target[:20]=source[:20]
    for index in range(length-21):
        start=4*index+20
        target[start]=source[index+20]
        for offset in (1,2,3):
            target[start+offset]=(4-offset)*source[index+20]/4+offset*source[index+21]/4
    for offset in range(4):
        target[-4+offset]=source[-1]+offset*(source[-1]-source[-2])/4
    return target.float()


def native_checks(model):
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(1491)
        images=torch.randn(2,3,224,224)
        tokens=longclip.tokenize(['a detailed picture of a city street','a person reading a document'],truncate=True)
        model.eval()
        z,hidden=model.encode_image(images,return_token_hidden=True)
        text,ht=model.encode_text(tokens,return_full=True)
        plain_z,plain_t=model.encode_image(images),model.encode_text(tokens)
        torch.testing.assert_close(z,plain_z,atol=0,rtol=0)
        torch.testing.assert_close(text,plain_t,atol=0,rtol=0)
        student,_,metadata=load_native_state(model.state_dict())
        sz,st=student.encode_image(images),student.encode_text(tokens)
        torch.testing.assert_close(z,sz,atol=0,rtol=0)
        torch.testing.assert_close(text,st,atol=0,rtol=0)
        return dict(passed=True,backbone=metadata,image_shape=list(z.shape),
                    hidden_shape=list(hidden.shape),text_hidden_shape=list(ht.shape),
                    image_interface_max_abs=float((z-plain_z).abs().max()),
                    text_interface_max_abs=float((text-plain_t).abs().max()),
                    native_export_image_max_abs=float((z-sz).abs().max()),
                    native_export_text_max_abs=float((text-st).abs().max()))


def main():
    torch.set_num_threads(4)
    RUN.mkdir(parents=True,exist_ok=True)
    (EXP/'evidence').mkdir(parents=True,exist_ok=True)
    cfg=json.loads(CONFIG.read_text())
    source=Path.home()/'.cache/clip/ViT-L-14.pt'
    assert file_sha(source)==L14_SHA256
    seed_all(0)
    clip,_=longclip.load_from_clip('ViT-L/14',device='cpu',args=argparse.Namespace())
    dimensions=validate_backbone(clip,'ViT-L/14')
    original=torch.jit.load(str(source),map_location='cpu').state_dict()
    for name,value in clip.state_dict().items():
        if name.startswith('mask_net.') or name in ('positional_embedding','positional_embedding_res'):
            continue
        torch.testing.assert_close(value,original[name].float(),atol=0,rtol=0,msg=name)
    positions=expected_positions(original['positional_embedding'])
    torch.testing.assert_close(clip.positional_embedding,positions,atol=0,rtol=0)
    torch.testing.assert_close(clip.positional_embedding_res,positions,atol=0,rtol=0)
    assert not clip.positional_embedding.requires_grad and clip.positional_embedding_res.requires_grad
    del original
    before=torch.get_rng_state().clone()
    module=BalancedSearch(clip,search_hparams=hparams(cfg),fusion='balanced_stack',visual='patch',
                          checkpoint_encoders=False,image_chunk=128,text_chunk=128)
    assert torch.equal(before,torch.get_rng_state())
    a=dict(module.clip.mask_net.resblocks.named_parameters())
    b=dict(module.fusion_branch.visual_blocks.named_parameters())
    assert a.keys()==b.keys()
    for name in a:
        assert a[name].data_ptr()!=b[name].data_ptr()
        torch.testing.assert_close(a[name],b[name],atol=0,rtol=0)
    assert module.fusion_branch.visual_adapter.weight.shape==(768,1024)
    assert module.fusion_branch.gate.weight.shape==(768,1536)
    assert module.fusion_branch.gate.bias is None and module.fusion_branch.gate.weight.count_nonzero()==0
    optimizer=build_optimizer(module)
    assert not optimizer.state
    module_summaries=dict(text_blocks=state_digest(module.clip.mask_net.resblocks.state_dict()),
                          visual_blocks=state_digest(module.fusion_branch.visual_blocks.state_dict()),
                          visual_adapter=state_digest(module.fusion_branch.visual_adapter.state_dict()),
                          gate=state_digest(module.fusion_branch.gate.state_dict()),
                          shared_pool=state_digest(module.clip.mask_net.attn_pool.state_dict()))
    real_l14=native_checks(clip)
    assert file_sha(B16_PATH)==B16_SHA
    bare=torch.load(B16_PATH,map_location='cpu',weights_only=True)
    legacy,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    legacy.load_state_dict(bare,strict=True)
    b16=native_checks(legacy)
    provenance=dict(source='OpenAI CLIP + original random MaskNetwork',base_model='ViT-L/14',
                    seed=0,input_resolution=224,context_length=248,pretrained_url=L14_URL,
                    pretrained_sha256=L14_SHA256,dimensions=dimensions,module_initialization=module_summaries,
                    optimizer_initial_state_empty=True,initialization_rng_isolated=True,
                    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                    loader_sha256=file_sha(REPO/'model/longclip.py'),
                    model_code_sha256=file_sha(REPO/'model/model_longclip.py'))
    payload=dict(model=clip.state_dict(),adapter=module.fusion_branch.state_dict(),optimizer=optimizer.state_dict(),
                 completed_steps=0,rng_cpu=before,config=cfg,provenance=provenance)
    INIT.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(payload,INIT)
    report=dict(passed=True,initial_checkpoint=str(INIT),initial_checkpoint_sha256=file_sha(INIT),
                provenance=provenance,real_l14_native=real_l14,b16_native_regression=b16,
                reference_commit=REFERENCE_COMMIT,reference_bare_sha256=B16_SHA,
                gate_zero_initialized=True,text_visual_independent=True,shared_pool_registered_once=True,
                position_extension_exact=True,cache_path_used=False)
    (RUN/'initialization.json').write_text(json.dumps(report,indent=2)+'\n')
    (EXP/'evidence/initialization.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
