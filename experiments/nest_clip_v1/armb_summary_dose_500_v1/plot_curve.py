"""Standalone plotting in an isolated environment; no model imports or updates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP=Path(__file__).resolve().parent

def main():
    data=json.loads((EXP/'PLOT_INPUT.json').read_text());curve=data['curve'];baseline=data['baseline'];keys=['Score5_R1','J_long3','J_long','Short4_R1']
    x=[r['S_weight'] for r in curve];fig,axes=plt.subplots(2,2,figsize=(10,6.5))
    for ax,key in zip(axes.flat,keys):
        y=[r['scores'][key]*100 for r in curve];ax.plot(x,y,'o-',color='#2463a0',linewidth=2)
        ax.axhline(baseline[key]*100,color='#666666',linestyle='--',label='RandomK matched500')
        ax.set_xticks(x);ax.invert_xaxis();ax.set_xlabel('Summary alignment weight (decreases to right)');ax.set_ylabel(key+' (%)');ax.grid(alpha=.2);ax.margins(x=.08,y=.12)
        for xx,yy in zip(x,y):ax.annotate(f'{yy:.3f}',(xx,yy),textcoords='offset points',xytext=(0,7),ha='center',fontsize=8)
        ax.legend(fontsize=8)
    fig.suptitle('Summary alignment dose: one seed, matched500 updates');fig.tight_layout()
    fig.savefig(EXP/'DOSE_CURVE.png',dpi=160);fig.savefig(EXP/'DOSE_CURVE.svg');plt.close(fig)
    svg=EXP/'DOSE_CURVE.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')

if __name__=='__main__':main()
