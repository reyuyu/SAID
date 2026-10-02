"""Real-sample equality and readable absolute-position evidence before training."""
import copy
import json
from pathlib import Path
import subprocess

import torch

from model import longclip
from train.nested_semantic_data import NestedDataset,file_sha
from experiments.nest_clip_v1.balanced_hparam_search_v1.search import SHARED,REPO


EXP=Path(__file__).resolve().parent
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/balanced_rmask_500_v1')


def main():
    torch.set_num_threads(4)
    assert file_sha(SHARED)=='54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'
    source=subprocess.check_output(['git','show','14653c92c6da9d552a2b624ab169eaaa275cdde8:model/nested_fusion_mask.py'],cwd=REPO)
    assert source==(REPO/'model/nested_fusion_mask.py').read_bytes()
    for filename in ('model/balanced_hparam_search.py','model/model_longclip.py'):
        expected=subprocess.check_output(['git','show','14653c92c6da9d552a2b624ab169eaaa275cdde8:'+filename],cwd=REPO)
        assert expected==(REPO/filename).read_bytes()
    datasets=[NestedDataset('/root/lk_projects/SAID-nest-clip-v1/data_index',
                            '/root/lk_projects/SAID-assets/training/ShareGPT4V','random_k',0,mode)
              for mode in ('compact','prefix_pad')]
    evidence=[];matched=0;valid=0
    for epoch in range(4):
        for dataset in datasets:dataset.set_epoch(epoch)
        for index in range(32):
            old,new=[dataset[index] for dataset in datasets]
            assert torch.equal(old['image'],new['image'])
            assert old['K']==new['K'] and old['n']==new['n'] and old['valid']==new['valid']
            assert torch.equal(old['tokens_f'],new['tokens_f']) and torch.equal(old['tokens_o'],new['tokens_o'])
            matched+=1
            if not new['valid']:
                assert torch.equal(old['tokens_e'],new['tokens_e'])
                continue
            valid+=1
            b=new['prefix_token_boundary'];e=new['full_eot_index']
            assert torch.equal(new['tokens_e'][b:],new['tokens_f'][b:])
            assert bool((new['tokens_e'][1:b]==0).all())
            assert int(new['tokens_e'].argmax())==int(new['tokens_f'].argmax())
            if len(evidence)<12:
                evidence.append(dict(sample_id=new['sample_id'],epoch=epoch,
                    sentences=new['views'][0].split('. '),K=new['K'],n=new['n'],
                    F_token_length=e+1,prefix_boundary=b,prefix_last_masked_position=b-1,
                    EOT_index=e,R_old_nonpad_positions=old['tokens_e'].nonzero().flatten().tolist(),
                    R_mask_nonpad_positions=new['tokens_e'].nonzero().flatten().tolist(),
                    suffix_ids=new['tokens_f'][b:e].tolist(),
                    suffix_original_positions=list(range(b,e))))
    # Real B16 encode_text accepts internal PAD while retaining the original EOT.
    initial=torch.load(SHARED,map_location='cpu',weights_only=False)
    model,_=longclip.load_from_clip('ViT-B/16',device='cpu')
    model.load_state_dict(initial['model'],strict=True)
    datasets[1].set_epoch(0)
    tokens=torch.stack([datasets[1][i]['tokens_e'] for i in range(4)])
    with torch.no_grad():
        embeddings=model.encode_text(tokens)
    assert embeddings.shape==(4,512) and torch.isfinite(embeddings).all()
    result=dict(passed=True,real_sample_epoch_pairs=matched,valid_pairs=valid,
                image_F_P_K_identical=True,suffix_ids_positions_identical=True,
                SOT_EOT_preserved=True,prefix_including_separator_all_PAD=True,
                real_b16_encode_text_finite=True,model_and_loss_source_unchanged=True,
                common_step0_sha256=file_sha(SHARED),readable_samples=evidence)
    (EXP/'evidence').mkdir(parents=True,exist_ok=True)
    (EXP/'evidence/real-samples.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='readable_samples'},indent=2))


if __name__=='__main__':main()
