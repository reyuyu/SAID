# Copy to your own untracked location and replace every CHANGE_ME value.
# Never change init path, worker count, LR or horizon between continuation stages.
export REPRO_BUNDLE=/CHANGE_ME/SAID-docs/experiments/s0_dualmask_masked_3epoch
export REPRO_BASE_REPO=/CHANGE_ME/SAID-base500
export REPRO_TRAIN_REPO=/CHANGE_ME/SAID-train
export REPRO_RUN=/CHANGE_ME/runs/masked_reproduction
export REPRO_INIT=/CHANGE_ME/assets/cvssl_initial.pt
export REPRO_PYTHON=/CHANGE_ME/envs/said-smartclip/bin/python
export SHARE4V_DATA_ROOT=/CHANGE_ME/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export REPRO_COCO_ROOT=/CHANGE_ME/datasets/coco
export REPRO_URBAN_ROOT=/CHANGE_ME/datasets/Urban1k/Urban1k
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export GLOO_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
