# Said attention visualization dashboard

Read-only Streamlit viewer for the Said-attention diagnostic artifacts produced
during training. The dashboard **never** loads a checkpoint and **never** runs the
model, so switching samples / checkpoints is instant.

## 1. Export artifacts (training side, needs a GPU)

```bash
cd /root/SAID
export CUDA_VISIBLE_DEVICES=0 \
       SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V \
       SHARE4V_JSON=debug/share4v_smoke_nosam.json

python eval/salu/export_dashboard_artifacts.py \
  --checkpoints "initial:,step100:runs_salu/phase22/salu_said_only_step000100.pt,step200:runs_salu/phase22/salu_said_only_step000200.pt,step400:runs_salu/phase22/salu_said_only_step000400.pt,final:runs_salu/phase22/salu_said_only_last.pt" \
  --num_samples 64 \
  --output_dir outputs/salu_dashboard
```

Artifacts (git-ignored):

Every checkpoint directory is self-contained: it contains every sample image
as well as its four attention maps. Images are the model-input crops, so the
14x14 maps align with them. The exporter writes images for every checkpoint,
including on a clean export; copying old images is not required.

The export-to-loader integration test uses two checkpoints and checks both
images and all caption attention variants:

```bash
python -m pytest tests/test_dashboard_export.py tests/test_said_dashboard.py -q
```

```text
outputs/salu_dashboard/
├── manifest.json              # checkpoints, 64 sample indices/image ids, captions, permutation
├── metrics.json               # per-checkpoint: entropy, JSD, z_s cosine, route/evidence acc+margin, precision noise, ratio
├── initial/ … final/
│   ├── sampleXX_image.png     # original image thumbnail (no training data committed)
│   ├── sampleXX_{own,shuffled,short,long}.npy   # 14x14 Said attention
│   └── per_sample_metrics.json
```

## 2. Install and start the dashboard

```bash
pip install -r tools/said_dashboard/requirements.txt

cd /root/SAID
streamlit run tools/said_dashboard/app.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

Loopback only by design (`127.0.0.1`), no credentials in the repo. Reach it from
your laptop with an SSH tunnel:

```bash
ssh -L 8501:127.0.0.1:8501 -p <port> root@<server>
# then open http://127.0.0.1:8501
```

If the artifacts live elsewhere, set `SAID_DASHBOARD_ROOT` (or type the path in the
sidebar).

## 3. What the dashboard shows

| panel | content |
| --- | --- |
| Image and captions | original image + the four captions (own / shuffled / short / long) |
| Same image / different caption | `Original · A · B · |A−B|` with a **shared** colour scale, plus JSD, mean abs diff, z_s cosine and the caption-conditioning ratio |
| Attention evolution | the same image + caption across all checkpoints, one shared colour scale |
| Metric evolution | entropy, effective patch count, caption JSD vs precision-noise JSD, route/evidence accuracy, conditioning ratio, z_s cosine |
| Precision noise panel | caption-change JSD vs bf16/fp32 noise JSD and their ratio |
| Route identification | route / evidence top-1 accuracy and margin (chance = 1/64) |

## 4. Notes

* Attention colour scale is shared by default; independent per-tile min-max
  normalisation would exaggerate tiny differences.
* `caption-conditioning ratio = caption JSD / precision-noise JSD`. Above 1 (ideally
  above 2) means the caption changes the attention more than numerical precision
  does. The noise floor itself depends on how sharp the attention is, so read it
  together with the absolute JSD values.
* The dashboard is diagnostic only: it never feeds back into training or loss.
