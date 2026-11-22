"""Look for additional short emergency routes in the official Master7sE loss and other public v6 games."""
import argparse
import json
from pathlib import Path
from branch_replay import ROOT,OUT,configure_engine
from benchmark import load_strategy,limit_math_threads
from starvation_replay import read_replay
from depot_corridor import _v7_corridor


def run(p,side,m):
    d=read_replay(p);hist={};prev=None;action={};events=[]
    for t in range(len(d['steps'])-1):
        f=d['steps'][t];o=dict(f[0]['observation'],player=side)
        drought=m._v7_watch_cash(o,prev,action,hist);prev=o['players']
        action=d['steps'][t+1][side]['action'] or {}
        new,route=_v7_corridor(o,d['configuration'],action,drought)
        if route:
            wid,home,length,net=route
            events.append(dict(turn=t,uid=wid,home=home,length=length,current_tree_adjusted=round(net,2),
                               original=action.get(wid),proposed=new[wid],changed=new!=action))
    return dict(file=p.name,seat=side,offers=events)


def main():
    pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('folder',type=Path);a=pa.parse_args()
    limit_math_threads();configure_engine();m=load_strategy('corridor_local',ROOT/'experiments/v7/checkpoints/jetty.py')
    _v7_corridor.__globals__.update({k:getattr(m,k) for k in ('_V7_L','_V7_D','WIDX','WACT','LAST_TURN')})
    rows=[run(a.folder.parent/'starvation-4063.html.gz',1,m)]
    for r in json.loads((a.folder/'index.json').read_text()):rows.append(run(a.folder/r['file'],r['seat'],m))
    (OUT/'depot-corridors.json').write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:
        if r['offers']:print(r['file'],len(r['offers']),[x['turn'] for x in r['offers'] if x['changed']][:16])
if __name__=='__main__':main()
