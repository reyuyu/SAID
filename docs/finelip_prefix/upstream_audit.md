# FP0 upstream preflight audit

Official checkout: `/root/external/FineLIP` (separate repository, remote `https://github.com/tiiuae/FineLIP.git`). Fixed commit: `2118312c9d640c71904379e90129649a46e6f2dd`; root tree: `d2c80bb10979321d8771e662fb48b165a419b750`; `train` tree: `44807719f37005b23039e78f1ca6a53bd4dfde57`.

The Git objects contain the requested training files. Blob IDs: `arguments.py` `efd63c0592a780ea937dbc33ccd7e4f421d8d1b0`, `loss.py` `d3131877b286e038106524d524fd3521420e7f77`, `scheduler.py` `7d83bdda016d0d5d7e699b618f94bca2e033aa82`, `train.py` `710e8b4b91c92b22cfe2ba6d9d5523eadfa1992d`, `test.py` `4cf20a1653cb41ae416b1d97e732716e3e5e5905`. The earlier statement that the fixed commit did not contain `train/` was incorrect in scope: the initial directory listing did not reflect the complete Git tree; object-level verification confirms the files.

No LICENSE file or explicit reuse permission was identified. Keep `LICENSE_NOT_IDENTIFIED` and `REUSE_PERMISSION_UNCONFIRMED`; do not copy upstream source into SAID until permission is established. FP0 may reference the APIs and use independently written adapter code.
