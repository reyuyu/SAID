# B formal run invalidated

The first formal launch was stopped at approximately update 79.  The
ablation subclass was selected correctly, but the initial controller revision
failed to pass `search_hparams` into the subclass, so its fusion-adapter peak
LR was the default `1e-4` rather than the required E2 `2e-4`.  No checkpoint
from this trajectory is used.  Its runtime directory and `steps.jsonl` remain
at:

`/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/said-b16-component-ablation-stop3651-v1/B/formal3651`

The valid run restarts from the immutable common step-0 checkpoint after the
controller fix.  The invalid trajectory is retained as engineering evidence.
