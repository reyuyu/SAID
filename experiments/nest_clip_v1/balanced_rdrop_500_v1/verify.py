"""Verified reuse of matched baseline and independently seeded sentence subsets."""
import json
from pathlib import Path
import subprocess

import torch

from model import longclip
from train.nested_semantic_data import NestedDataset,file_sha
from experiments.nest_clip_v1.balanced_hparam_search_v1.search import REPO,SHARED,load


EXP=Path(__file__).resolve().parent
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/balanced_rdrop_500_v1')
BASELINE_STATE=Path('/root/lk_projects/SAID-nest-clip-v1/balanced_rmask_500_v1/state.json')
BASELINE_SHA='f44e3ab0566a12e189514a92fedac3416299f87ca129b649a2c76a63b222b3f8'


def main():
    torch.set_num_threads(4)
    baseline=load(BASELINE_STATE)['roles']['baseline']['result']
    assert baseline['checkpoint_sha256']==BASELINE_SHA and file_sha(baseline['checkpoint'])==BASELINE_SHA
    assert baseline['config']['horizon']==4868 and baseline['config']['max_updates']==500
    assert baseline['config']['init_sha256']==file_sha(SHARED)
    for filename in ('model/nested_fusion_mask.py','model/balanced_hparam_search.py','model/model_longclip.py'):
        source=subprocess.check_output(['git','show','fa19d12:'+filename],cwd=REPO)
        assert source==(REPO/filename).read_bytes()
    old=NestedDataset('/root/lk_projects/SAID-nest-clip-v1/data_index',
                      '/root/lk_projects/SAID-assets/training/ShareGPT4V','random_k',0,'compact')
    drop=NestedDataset(old.index_dir,old.image_root,'random_k',0,'sentence_drop')
    count=0;proofs=[];valid=0;same=0
    for epoch in range(4):
        old.set_epoch(epoch);drop.set_epoch(epoch)
        for index in range(32):
            a,b=old[index],drop[index]
            assert torch.equal(a['image'],b['image'])
            assert torch.equal(a['tokens_f'],b['tokens_f']) and torch.equal(a['tokens_o'],b['tokens_o'])
            assert a['K']==b['K'] and a['n']==b['n'] and a['valid']==b['valid']
            count+=1
            if not b['valid']:
                assert torch.equal(a['tokens_e'],b['tokens_e']);continue
            valid+=1
            k=b['K'];sentences=a['views'][0].split('. ');suffix=sentences[k:]
            q=b['drop_q'];indices=b['drop_selected_indices_after_sort']
            assert 1<=q<=len(suffix) and indices==sorted(set(indices)) and len(indices)==q
            assert b['views'][2]=='. '.join(suffix[i] for i in indices)
            expected=longclip.tokenize([b['views'][2]],context_length=248,truncate=False)[0]
            assert torch.equal(b['tokens_e'],expected)
            if q==len(suffix):assert torch.equal(a['tokens_e'],b['tokens_e'])
            same+=int(b['drop_same_old_r'])
            replay=drop[index]
            assert torch.equal(replay['tokens_e'],b['tokens_e']) and replay['drop_selected_indices_before_sort']==b['drop_selected_indices_before_sort']
            if len(proofs)<16:
                proofs.append(dict(sample_id=b['sample_id'],epoch=epoch,sentences=sentences,K=k,
                    old_suffix_sentences=suffix,m=len(suffix),q=q,index_base=1,
                    selected_indices_before_sort=[k+i+1 for i in b['drop_selected_indices_before_sort']],
                    selected_indices_after_sort=[k+i+1 for i in indices],
                    R_old_text=a['views'][2],R_drop_text=b['views'][2],
                    token_lengths=dict(F=int(b['tokens_f'].argmax())+1,P=int(b['tokens_o'].argmax())+1,
                                       R_old=int(a['tokens_e'].argmax())+1,R_drop=int(b['tokens_e'].argmax())+1)))
    model,_=longclip.load_from_clip('ViT-B/16',device='cpu')
    model.load_state_dict(torch.load(SHARED,map_location='cpu',weights_only=False)['model'],strict=True)
    drop.set_epoch(0)
    with torch.no_grad():encoded=model.encode_text(torch.stack([drop[i]['tokens_e'] for i in range(4)]))
    assert encoded.shape==(4,512) and torch.isfinite(encoded).all()
    result=dict(passed=True,matched_sample_epoch_pairs=count,valid_pairs=valid,same_old_r_samples=same,
                image_F_P_K_equal=True,ordered_whole_suffix_sentences_only=True,
                native_text_finite=True,baseline_reused=True,baseline_sha256=BASELINE_SHA,
                initial_sha256=file_sha(SHARED),model_loss_source_unchanged=True,readable_samples=proofs)
    (EXP/'evidence').mkdir(parents=True,exist_ok=True)
    (EXP/'evidence/real-samples.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='readable_samples'},indent=2))


if __name__=='__main__':main()
