"""Optional read-only F/D mask IoU; returns the original loss-path tensors."""
import torch

from model.nested_semantic_mask import global_sum


def install():
    import model.balanced_hparam_search as target
    original = target.fusion_view_terms
    original_forward = target.BalancedSearch.forward
    masks = []
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        diagnostics = args[9] if len(args)>9 else kwargs.get('diagnostics', False)
        if diagnostics:
            masks.append(result[2].detach())
            if len(masks)==3:
                f, d = masks[0], masks[2]
                valid, valid_global = args[5], args[6]
                with torch.no_grad():
                    iou = (f*d).sum(-1)/((f+d)>0).sum(-1).clamp_min(1)
                    value = global_sum(iou[valid].sum())/int(valid_global.sum())
                result[-1]['F_D_positive_mask_iou'] = value
                masks.clear()
        else:
            masks.clear()
        return result
    target.fusion_view_terms = observe
    def forward(module, *args, **kwargs):
        masks.clear()  # Includes global0/1 local-view fallbacks.
        return original_forward(module, *args, **kwargs)
    target.BalancedSearch.forward = forward
    def restore():
        target.fusion_view_terms = original
        target.BalancedSearch.forward = original_forward
    return restore
