"""Reconstruct phases of the two referenced games from official replays and event ledgers."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def distance(a,b):
    dr,dc=abs(a//15-b//15),abs(a%15-b%15)
    return min(dr,15-dr)+min(dc,15-dc)


def measure(replay,summary):
    states=[s[0]['observation'] for s in replay['steps']]
    events=summary['events']
    assert len(events)==summary['total_events'] and len(states)==300
    phases=[]
    for first,last in ((0,50),(50,100),(100,250),(150,250),(250,270),(270,299)):
        rows=[]
        for side in (0,1):
            count=Counter(e[1] for e in events if first<e[0]<=last and e[2]==side)
            deposits=sum(e[3] for e in events if first<e[0]<=last and e[1]=='deposit' and e[5]==side)
            income=sum(sum(o['trees'][tid][1] for tid in o['players'][side][3]) for o in states[first+1:last+1])
            intrusion=sum(any(k=='COLLECTOR' and v<30 and min(distance(p,q) for q in o['players'][1-side][1].values())<=2
                              for p,v,k in o['players'][side][2].values()) for o in states[first+1:last+1])
            rows.append(dict(events=dict(count),deposits=round(deposits,2),tree_income=round(income,2),
                             turns_light_collector_near_enemy_base=intrusion))
        phases.append(dict(start_exclusive=first,end_inclusive=last,players=rows))
    def money(o,side):return round(o['players'][side][0],2)
    snapshots=[]
    for t in (50,100,138,150,175,200,240,250,260,270,276,280,285,286,290,299):
        o=states[t]
        snapshots.append(dict(step=t,players=[dict(cash=money(o,side),workers=len(p[2]),
                              collectors=sum(k=='COLLECTOR' for _,_,k in p[2].values()),
                              carried=round(sum(v for _,v,_ in p[2].values()),2),trees=len(p[3]))
                              for side,p in enumerate(o['players'])]))
    return dict(order=summary['metadata']['order'],strategies=summary['metadata']['strategies'],phases=phases,
                snapshots=snapshots,scores=summary['metadata']['scores'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('v4_replay',type=Path);p.add_argument('master_replay',type=Path)
    p.add_argument('summaries',type=Path);p.add_argument('output',type=Path);p.add_argument('--plot',type=Path)
    a=p.parse_args();result={};datasets=[]
    for label,f in [('yan-v4',a.v4_replay),('Master7sD',a.master_replay)]:
        replay=json.loads(gzip.decompress(f.read_bytes()))
        summary=json.loads((a.summaries/f.name.removesuffix('.gz')).read_text())
        result[label]=measure(replay,summary);datasets.append(replay['steps'])
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(2,2,figsize=(10.3,6.7),sharex=True)
        for col,(frames,labels) in enumerate(zip(datasets,[('1C-Testv2','yan-v4'),('Master7sD','Xiaofeiyu v5')])):
            for side,name in enumerate(labels):
                obs=[f[0]['observation'] for f in frames];color=['#006a70','#c15123'][side]
                cash=[x['players'][side][0] for x in obs]
                income=[sum(x['trees'][k][1] for k in x['players'][side][3]) for x in obs]
                smooth=[sum(income[max(0,i-9):i+1])/min(10,i+1) for i in range(len(income))]
                axs[0,col].plot(cash,color=color,label=name,lw=2);axs[1,col].plot(smooth,color=color,lw=1.7)
            axs[0,col].legend(frameon=False,loc='upper left');axs[1,col].set_xlabel('Turn')
            for ax in axs[:,col]:
                ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False);ax.set_xlim(0,299)
        axs[0,0].set_ylabel('Cash');axs[1,0].set_ylabel('Tree income per turn\n(10-turn mean)')
        axs[0,0].axvspan(270,299,alpha=.08,color='#006a70')
        axs[0,1].axvspan(138,285,alpha=.08,color='#c15123')
        axs[0,1].annotate('19.28 cash for 148 turns',xy=(203,19.28),xytext=(118,1700),
                          arrowprops={'arrowstyle':'->','color':'#78351b'},color='#78351b',fontsize=9)
        fig.tight_layout();fig.savefig(a.plot,dpi=155);plt.close(fig)


if __name__=='__main__':
    main()
