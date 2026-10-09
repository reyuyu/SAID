"""HNS-only invocation of the archived five-state real/CPU DDP equivalence."""
import argparse
from recovery import hns_balanced_macro_equivalence as protocol


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config');p.add_argument('--real',action='store_true')
    args=p.parse_args()
    protocol.BRANCHES={'HNS':'origin/experiment/nested-d3-hard-nested-sparsity500-v1'}
    if args.real:protocol.real(args.output,[args.config])
    else:protocol.ddp(args.output)


if __name__=='__main__':main()
