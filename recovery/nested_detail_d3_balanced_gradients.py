"""Same frozen eight-batch D3 gradient spot-check with actual balanced weights."""
from recovery import nested_detail_gradients as protocol
from recovery.nested_detail_d3_gradients import D3LocalDataset
from recovery.nested_detail_d3_balanced500 import RUN,EXP,WEIGHTS


def main():
    protocol.RUN,protocol.EXP,protocol.WEIGHTS=RUN,EXP,WEIGHTS
    protocol.FullLocalDataset=D3LocalDataset
    protocol.main()


if __name__=='__main__':main()
