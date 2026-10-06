"""Unchanged eight-batch gradient protocol with D3 in the lowest input slot."""
from recovery import nested_detail_gradients as protocol
from recovery.nested_detail_d3_equal500 import RUN,EXP,WEIGHTS,MODE
from recovery.s02_full_local_data import FullLocalDataset


class D3LocalDataset(FullLocalDataset):
    def __init__(self,index,images,unused_mode,seed):
        super().__init__(index,images,MODE,seed)


def main():
    protocol.RUN,protocol.EXP,protocol.WEIGHTS=RUN,EXP,WEIGHTS
    protocol.FullLocalDataset=D3LocalDataset
    protocol.main()


if __name__=='__main__':main()
