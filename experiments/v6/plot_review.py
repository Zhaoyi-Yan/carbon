"""Plot the opening cash and tree-income gaps for all 59 completed online games."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('review',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();groups=json.loads(a.review.read_text())['outcome_groups']
    font=Path('/System/Library/Fonts/Hiragino Sans GB.ttc')
    zh=FontProperties(fname=str(font)) if font.exists() else None
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharex=True)
    for ax,label,key in zip(axes,('现金差','累计树收入差'),('cash','absorption')):
        ax.axhline(0,color='#89929a',linewidth=.8)
        for group,name,color in (('wins','最终获胜（42 局）','#177965'),('losses','最终落败（17 局）','#b44332')):
            by=groups[group]['by_step']
            x=[0]+[int(t) for t in by if int(t)<=100]
            y=[0]+[by[str(t)]['median_margin'][key] for t in x[1:]]
            ax.plot(x,y,'o-',color=color,linewidth=2,markersize=3.5,label=name if zh else group)
        ax.set_title(label if zh else key,fontproperties=zh,loc='left',fontsize=13)
        ax.set_xlim(0,100);ax.set_xlabel('回合' if zh else 'Turn',fontproperties=zh)
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('yan-v5 减对手，各局中位数' if zh else 'Median v5 minus rival',fontproperties=zh)
    axes[0].legend(frameon=False,prop=zh,loc='lower left')
    fig.tight_layout(pad=1.5)
    fig.savefig(a.output,dpi=155,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
