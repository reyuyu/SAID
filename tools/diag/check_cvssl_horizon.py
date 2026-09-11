"""Pre-flight: the LR horizon must be 3 x the REAL len(train_loader), not 3 x a guessed value."""
import os
import sys

import torch

REPO = '/root/SAID-gap-completion'
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + '/train')

from said_cvssl_data import Share4VCvsslDataset, cvssl_collate  # noqa: E402

BATCH = 256
WORLD = 4
EPOCHS = 3


def main():
    dataset = Share4VCvsslDataset(seed=0, augment_view_b=True,
                                  strict_manifest=os.environ.get('SHARE4V_FULL_AUDIT'))
    # the trainer builds DistributedSampler(dataset) and DataLoader(batch_size=256, drop_last=False),
    # and DistributedSampler.__len__ is ceil(len(dataset) / num_replicas) on this torch version,
    # so len(loader) == ceil(len(dataset) / batch_size) regardless of the world size
    steps_per_epoch = -(-len(dataset) // BATCH)          # ceil
    print('DATASET_SAMPLES %d' % len(dataset))
    print('BATCH_PER_RANK %d WORLD %d GLOBAL_PAIRS %d GLOBAL_VIEWS %d'
          % (BATCH, WORLD, BATCH * WORLD, BATCH * WORLD * 2))
    print('STEPS_PER_EPOCH %d' % steps_per_epoch)
    print('LR_HORIZON 3 * steps_per_epoch = %d' % (EPOCHS * steps_per_epoch))
    print('NOTE the loader has no drop_last, so the last batch of an epoch is ragged '
          '(<= %d per rank); the trainer supports it' % BATCH)


if __name__ == '__main__':
    main()
