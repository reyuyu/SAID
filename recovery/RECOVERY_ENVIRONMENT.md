# Recovery environment

Sources: pinned `docs/environment_freeze.txt`, `requirements.txt`, and
`experiments/nest_clip_v1/evidence/environment.json`. No core dependency upgrades.

| Item | Historical | Recovered |
| --- | --- | --- |
| Python | 3.10, patch release not established | 3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0] |
| CUDA wheel runtime | 12.4 | 12.4 |
| NCCL | 2.21.5 | [2, 21, 5] |
| Visible GPUs | 4 x A100 80GB | 4 |
| absl-py | 2.5.0 | 2.5.0 |
| filelock | 3.32.3 | 3.32.3 |
| fsspec | 2026.7.0 | 2026.7.0 |
| ftfy | 6.3.1 | 6.3.1 |
| grpcio | 1.83.1 | 1.83.1 |
| Jinja2 | 3.1.6 | 3.1.6 |
| Markdown | 3.10.3 | 3.10.3 |
| MarkupSafe | 3.0.3 | 3.0.3 |
| mpmath | 1.3.0 | 1.3.0 |
| networkx | 3.4.2 | 3.4.2 |
| numpy | 2.2.6 | 2.2.6 |
| nvidia-cublas-cu12 | 12.4.5.8 | 12.4.5.8 |
| nvidia-cuda-cupti-cu12 | 12.4.127 | 12.4.127 |
| nvidia-cuda-nvrtc-cu12 | 12.4.127 | 12.4.127 |
| nvidia-cuda-runtime-cu12 | 12.4.127 | 12.4.127 |
| nvidia-cudnn-cu12 | 9.1.0.70 | 9.1.0.70 |
| nvidia-cufft-cu12 | 11.2.1.3 | 11.2.1.3 |
| nvidia-curand-cu12 | 10.3.5.147 | 10.3.5.147 |
| nvidia-cusolver-cu12 | 11.6.1.9 | 11.6.1.9 |
| nvidia-cusparse-cu12 | 12.3.1.170 | 12.3.1.170 |
| nvidia-nccl-cu12 | 2.21.5 | 2.21.5 |
| nvidia-nvjitlink-cu12 | 12.4.127 | 12.4.127 |
| nvidia-nvtx-cu12 | 12.4.127 | 12.4.127 |
| opencv-python-headless | 5.0.0.93 | 5.0.0.93 |
| packaging | 26.3 | 26.3 |
| pillow | 12.3.0 | 12.3.0 |
| protobuf | 7.36.1 | 7.36.1 |
| pycocotools | 2.0.11 | 2.0.11 |
| regex | 2026.9.3 | 2026.9.3 |
| sympy | 1.13.1 | 1.13.1 |
| tensorboard | 2.21.0 | 2.21.0 |
| tensorboard-data-server | 0.7.2 | 0.7.2 |
| torch | 2.5.1+cu124 | 2.5.1+cu124 |
| torchvision | 0.20.1+cu124 | 0.20.1+cu124 |
| tqdm | 4.70.0 | 4.70.0 |
| triton | 3.1.0 | 3.1.0 |
| typing_extensions | 4.16.0 | 4.16.0 |
| wcwidth | 0.8.3 | 0.8.3 |
| Werkzeug | 3.1.8 | 3.1.8 |

The isolated `.venv` uses the existing Python3.10.20 interpreter at
`/root/miniconda3/envs/RoboTwin/bin/python`; no packages in that environment were modified.
Its base interpreter must remain available. Full resolved versions are preserved in
`evidence/recovered-pip-freeze.txt`.

Unrecorded historic versions: Python patch release, pip, pytest, and exact setuptools
(original requirement is setuptools<81). Recovery uses pip25.2, pytest8.4.2, setuptools80.9.0.
pip was replaced to fix wheel-metadata name normalization, not to upgrade the training stack.
The system's default CUDA toolkit symlink is 12.2; PyTorch uses its pinned cu124 wheels.
Original driver and current driver are 550.127.05. No system CUDA/driver upgrade was performed.

Environment audit passed: True.

## Optional ModelScope recovery reader

SA1B-Paired-Captions-Images metadata recovery adds `pyarrow==21.0.0` only to
the isolated `.download-venv`, installed with `--no-deps`. The pinned training
`.venv`, PyTorch, torchvision, CUDA, and existing Hugging Face/OpenDataLab
download dependencies are not upgraded. The reader uses the official public
ModelScope repository API plus HTTP ranges for parquet footer/full-image URL
columns; it does not install ModelScope into the training environment or fetch
the full parquet/image dataset. Exact RGB/native-transform comparisons and
required-image installations run in the unchanged training `.venv` on CPU.
