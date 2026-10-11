# Reproduction and provenance

The released identity is HNS-v1 ViT-B/16, step 3651, from planned four-epoch horizon 4868. Source commit: `45468e1bbbcee52f6c069db79ee8266d529ca7a`; training branch: `experiment/nested-d3-hns-full-v1`; evaluation branches: `analysis/d3-hns-epoch-curve-eval-v1` and `experiment/hns-v1-step3651-flickrfull-eval-v1`.

The training checkpoint is a PyTorch pickle-style artifact. Load it only from a trusted source after verifying its SHA256. It may contain paths and Python objects; the release audit must remove credentials and private service metadata before any redistribution. Never upload raw training images or credentials.
