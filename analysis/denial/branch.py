"""Replay one action from a real situation while the other side takes its recorded move."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark import configure_engine,limit_math_threads,load_strategy


def stats(board,side):
    p=board['players'][side]
    return dict(cash=round(p[0],2),collectors=sum(k=='COLLECTOR' for _,_,k in p[2].values()),
                carried=round(sum(v for _,v,_ in p[2].values()),3),trees=len(p[3]))


def run(model,path,frames,conf,side,turns):
    m=load_strategy('denial_branch_'+model,path);rows=[]
    for t,(before,after) in enumerate(zip(frames,frames[1:])):
        if t>max(turns):break
        obs=before[0]['observation'];pl=obs['players']
        public=dict(step=t,player=side,players=pl,trees=obs['trees'],carbon=before[side]['observation']['carbon'])
        start=time.perf_counter();action=m._act(public,conf if t in turns else None,start)
        original=[v['action'] or {} for v in after]
        if t in turns:
            # Privileged carbon only enters the offline referee after the submitted actions are fixed.
            board=dict(step=t,player=side,players=pl,carbon=obs['full_carbon'],trees=obs['trees'])
            fixed=m._turn(board,conf,original);live=after[0]['observation']
            for who in (0,1):
                a,b=stats(fixed,who),stats(live,who)
                assert abs(a['cash']-b['cash'])<=.02 and abs(a['carried']-b['carried'])<.05 and a['trees']==b['trees'] and a['collectors']==b['collectors'],(t,who,a,b)
            alternative=original.copy();alternative[side]=action
            out=m._turn(board,conf,alternative)
            rows.append(dict(step=t,next_step=t+1,side=side,model=model,actual=[stats(live,i) for i in (0,1)],
                             counterfactual=[stats(out,i) for i in (0,1)],action=action,original=original[side]))
        m._ST['tr'].set_actions(pl,original[side])
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('v4_game','master_game','parent','candidate','output'):p.add_argument(k,type=Path)
    a=p.parse_args();configure_engine();limit_math_threads();out=[]
    for file,side,turns in ((a.master_game,1,[134,140]),(a.master_game,0,[285]),(a.v4_game,1,[174,199,239,269])):
        r=json.loads(gzip.decompress(file.read_bytes()))
        for label,path in (('multistyle',a.parent),('denial',a.candidate)):
            for row in run(label,path.resolve(),r['steps'],r['configuration'],side,set(turns)):
                row['game']=file.name.split('.')[0];out.append(row)
    a.output.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    main()
