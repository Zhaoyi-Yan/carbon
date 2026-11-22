"""Add predicted and realized bilateral deposits to the already frozen paired proposal study."""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
from antisym_data import PARENT,ROOT,configure_engine,load,prepare,manifest,stockroot,expected,margin
from receipt_data import LAND,ACT

def bank(pl,orders):
    homes={p:s for s,team in enumerate(pl) for p in team[1].values()}
    totals=[0.,0.]
    for s,team in enumerate(pl):
        for u,(pos,cargo,role) in team[2].items():
            if not cargo:continue
            where=LAND[pos][ACT[orders[s].get(u) or 'NONE']]
            owner=homes.get(where)
            if owner is not None:totals[owner]+=cargo
    return totals

def main(data):
    configure_engine();a=data/'antisym-branch';out=data/'cutflow-roots';out.mkdir(exist_ok=True)
    originals=manifest();m=load(PARENT)
    # Keep identical birth/death/waypoint handling to the stored 35-step continuations.
    source=inspect.getsource(expected).replace('    for j in range(35):','    for j in range(11):')
    scope=expected.__globals__.copy();exec(source,scope);eleven=scope['expected']
    for meta,r in originals:
        path=a/(meta['name']+'.json')
        if not path.exists():continue
        dest=out/path.name
        if dest.exists():continue
        archived=json.loads(path.read_text());assert archived['pool_revision']==3
        turns={v['t']:v for v in archived['rows']}
        cfg,states,observed=prepare(m,meta,r);s=meta['side']
        m._ST.clear();m._FC_HISTORY.clear();tracker=m.BeliefTracker(m._Bel,s);prev=None;records=[]
        for t in range(max(turns,default=-1)+1):
            o=observed[t];pl=o['players'];actual=r['actions_and_post_cash'][t][0]
            pos=stockroot(m,o,cfg,prev,tracker,t in turns)
            if pos:
                group=turns[t];root,_,_=pos;features=[];outcomes=[]
                for i,move in enumerate(group['actions']):
                    orders=list(actual);orders[s]=move
                    stage=m._turn(root,cfg,orders)
                    ff=m._v7_flows(stage)
                    old_asset=m._r6_asset_score(stage,s)*3000.
                    startcash=margin(stage['players'],s);stowed=bank(pl,orders)
                    features.append([float(ff[s,0]-ff[1-s,0]),float(ff[s,1]-ff[1-s,1]),float(startcash),float(old_asset),
                                     float(stowed[s]-stowed[1-s]),ff.tolist()])
                    yy=[]
                    for mode in (0,1):
                        saved=m._turn;arr=[]
                        def traced(b,config,cmd):
                            arr.append(bank(b['players'],cmd))
                            return saved(b,config,cmd)
                        m._turn=traced
                        try:
                            check=eleven(m,states[t],t,s,move,states,r['actions_and_post_cash'],cfg,bool(mode))
                        finally:m._turn=saved
                        assert abs(check[0][0]-group['y'][mode][i][0][0])<.015
                        totals=np.zeros((2,2))
                        for k,h in enumerate((5,10)):
                            if len(arr)>1:totals[:,k]=np.asarray(arr[1:1+h]).sum(0)
                        yy.append(totals.round(4).tolist())
                    outcomes.append(yy)
                records.append(dict(t=t,features=features,observed=outcomes,p=group['p'],y=group['y']))
            tracker.set_actions(pl,actual[s]);prev=pl
        dest.write_text(json.dumps(dict(**meta,rows=records),separators=(',',':'))+'\n')
        print(meta['name'],len(records),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);main(p.parse_args().data)
