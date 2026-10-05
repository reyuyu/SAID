"""Construct real pinned training modules on CPU without forward/backward or updates."""

import argparse
import math
import subprocess
import sys

from audit import EVIDENCE, PINS, RECOVERY, ROOT, RUNTIME, dump, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", choices=["randomk", "summary02"])
    args = parser.parse_args()
    source = ROOT / "worktrees/randomk" if args.configuration == "randomk" else ROOT
    role = "canonical" if args.configuration == "randomk" else "research"
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"]).decode().strip()
    assert head == PINS[role][1]
    sys.path.insert(0, str(source))
    import torch
    from model import longclip
    from model.balanced_hparam_search import BalancedSearch, hparams
    from train.train_nested_semantic_mask import build_optimizer, seed_all, state_digest, training_horizon

    torch.set_num_threads(8)
    configuration = load(RECOVERY / "configs" / (args.configuration + ".json"))
    seed_all(configuration["seed"])
    clip, _ = longclip.load_from_clip("ViT-B/16", device="cpu", args=argparse.Namespace())
    initial = torch.load(RUNTIME / "shared/step000000.pt", map_location="cpu", weights_only=False)
    clip.load_state_dict(initial["model"], strict=True)
    module = BalancedSearch(
        clip.float(), arm=configuration["arm"],
        checkpoint_encoders=configuration["checkpoint_encoders"],
        image_chunk=configuration["image_chunk"], text_chunk=configuration["text_chunk"],
        condition_mode=configuration["condition_mode"], shuffle_seed=configuration["shuffle_seed"],
        checkpoint_pair_blocks=configuration["checkpoint_pair_blocks"],
        fusion=configuration["fusion"], visual=configuration["visual"],
        search_hparams=hparams(configuration))
    optimizer = build_optimizer(module)
    assert not optimizer.state
    assert state_digest(module.clip.state_dict()) == state_digest(initial["model"])
    assert module.clip.context_length == 248
    assert all(parameter.device.type == "cpu" for parameter in module.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in module.parameters())
    batches = math.ceil(math.ceil(1245901 / configuration["world_size"]) / configuration["batch_size"])
    assert batches == 1217 and training_horizon(configuration, batches) == 4868
    groups = [dict(name=group["name"], lr=group["lr"], weight_decay=group["weight_decay"],
                   parameter_tensors=len(group["params"]),
                   parameters=sum(parameter.numel() for parameter in group["params"]))
              for group in optimizer.param_groups]
    assert [group["lr"] for group in groups] == [1e-6, 1e-3, 1e-3, 2e-4]
    dump(EVIDENCE / ("construction-" + args.configuration + ".json"), dict(
        passed=True, source_commit=head, cpu_only=True, optimizer_updates=0,
        context_length=248, sampling_mode=configuration["sampling_mode"],
        view_weights=module.search_hparams["view_weights"],
        horizon=4868, updates_per_epoch=batches, optimizer_state_empty=True,
        clip_initializer_unchanged=True, optimizer_groups=groups,
        model_state_sha256=state_digest(module.state_dict())))


if __name__ == "__main__":
    main()
