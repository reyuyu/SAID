"""Local positive sibling relation; no new masks, encoders or CE candidates."""
from collections import deque
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def sibling_terms(images, prefix, remainder, prefix_mask, remainder_mask):
    z, p, r = images.float(), prefix.float(), remainder.float()
    mp, mr = prefix_mask.float(), remainder_mask.float()
    tp = F.normalize(p, dim=-1, eps=1e-6)
    tr = F.normalize(r, dim=-1, eps=1e-6)
    pp = (F.normalize(z*mp, dim=-1, eps=1e-6)*tp).sum(-1)
    rp = (F.normalize(z*mr.detach(), dim=-1, eps=1e-6)*tp).sum(-1)
    rr = (F.normalize(z*mr, dim=-1, eps=1e-6)*tr).sum(-1)
    pr = (F.normalize(z*mp.detach(), dim=-1, eps=1e-6)*tr).sum(-1)
    pv, rv = F.relu(rp-pp), F.relu(pr-rr)
    return .5*(pv+rv), dict(c_PP=pp, c_RP=rp, c_RR=rr, c_PR=pr,
        P_violation=pv, R_violation=rv)


class RelationCollapseMonitor:
    """Frozen diagnostic-only guard; never changes coefficients or sample validity."""
    def __init__(self, config):
        self.window = int(config.get('relation_guard_window',25))
        self.patience = int(config.get('relation_guard_patience',8))
        self.relative_iou = float(config.get('relation_iou_fraction_baseline',.25))
        self.iou_floor = float(config.get('relation_iou_floor',.15))
        self.keep_min = float(config.get('relation_min_keep',.02))
        self.keep_max = float(config.get('relation_max_keep',.995))
        self.history = deque(maxlen=self.window);self.consecutive = 0
        self.reference = {}
        with Path(config['relation_baseline_steps']).open() as handle:
            for line in handle:
                row = json.loads(line);self.reference[row['step']] = row['oe_iou']
        self.last = None

    def update(self, step, logs):
        self.history.append({key:float(logs[key]) for key in (
            'oe_iou','F_positive_keep_ratio','O_positive_keep_ratio','E_positive_keep_ratio')})
        if len(self.history)<self.window:return None
        mean = {key:sum(row[key] for row in self.history)/self.window for key in self.history[0]}
        baseline = sum(self.reference[s] for s in range(step-self.window+1,step+1))/self.window
        threshold = max(self.iou_floor,self.relative_iou*baseline)
        reasons = []
        if mean['oe_iou']<threshold:reasons.append('P/R IoU collapsed relative to matched baseline')
        for key in ('F_positive_keep_ratio','O_positive_keep_ratio','E_positive_keep_ratio'):
            if not self.keep_min<mean[key]<self.keep_max:reasons.append(key+' degenerated')
        self.consecutive = self.consecutive+1 if reasons else 0
        self.last = dict(step=step,window=self.window,window_means=mean,baseline_iou=baseline,
                         iou_threshold=threshold,consecutive_bad_windows=self.consecutive,reasons=reasons)
        return '; '.join(reasons) if self.consecutive>=self.patience else None
