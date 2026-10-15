# Recommended checkpoints

| Backbone | Score5 checkpoint | Score5 | Long-text checkpoint | J_long3 | J_long | Recommendation |
|---|---:|---:|---:|---:|---:|---|
| ViT-B/16 | 3651 | 73.774418 | 3651 | 78.188029 | 87.040003 | step3651 |
| ViT-L/14 | 4651 | 78.003728 | 6085 | 82.423184 | 90.775002 | step4651 for Score5/J_long3; step6085 for J_long |

The selection is exploratory and uses repeatedly observed public benchmarks; it is not a held-out validation choice.

```json
{
  "selection_policy": "Score5 primary; J_long3, J_long, Urban, Flickr30k-Full, and health as secondary evidence",
  "exploratory_public_test_selection": true,
  "ViT-B/16": {
    "score5": {
      "step": 3651,
      "checkpoint_sha256": "83a47256442547a24b91f2be18e61c33d28fe252eee0106e79e104a57c722e07",
      "score5": 73.77441765717163
    },
    "long_text": {
      "step": 3651,
      "J_long3": 78.18802942861939,
      "J_long": 87.04000257492066
    },
    "recommended": {
      "step": 3651,
      "reason": "best Score5 and all secondary summary metrics"
    }
  },
  "ViT-L/14": {
    "score5": {
      "step": 4651,
      "checkpoint_sha256": "9a773bfb54e3dde33ae724e4e3a13a0614381050ac5b9aa29939d54198c1639b",
      "score5": 78.00372821672931
    },
    "long_text": {
      "step": 6085,
      "J_long3": 82.42318392634591,
      "J_long": 90.77500214576722
    },
    "recommended": {
      "step": 4651,
      "reason": "best Score5 and J_long3; step 6085 retained as J_long specialist"
    }
  }
}
```
