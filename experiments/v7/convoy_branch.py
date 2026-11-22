"""Attribute turns 190–210 of archived counterfactual command logs to individual workers."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from branch_replay import OUT,SNAPSHOT,restore,step,configure_engine
from close_replay import referee_events
from receipt_data import ACT,LAND

def dist(p,q):
    dy,dx=abs(p//15-q//15),abs(p%15-q%15)
    return min(dy,15-dy)+min(dx,15-dx)

def main():
    configure_engine();checkpoint=json.loads(SNAPSHOT.read_text());result={}
    for prefix in ('branch','branch-varied'):
        logs=json.load(gzip.open(OUT/(prefix+'-actions.json.gz')))
        past={r['key']:r for r in json.loads((OUT/(prefix+'-results.json')).read_text())['results']}
        pairs=[]
        for key in logs:
            if not key.endswith('-bank'):continue
            alt=key[:-4]+'control';state,env=restore(checkpoint['snapshot'],checkpoint['configuration'])
            pl=state[0].observation.players
            edits=[]
            for uid,(pos,cargo,kind) in pl[0][2].items():
                old=(logs[alt][0][0].get(uid) or 'NONE');new=(logs[key][0][0].get(uid) or 'NONE')
                if old==new:continue
                home=tuple(pl[0][1].values())
                edits.append(dict(uid=uid,cargo=cargo,pos=pos,original=old,bank=new,
                                  home_distance=[min(dist(LAND[pos][ACT[a]],h) for h in home) for a in (old,new)]))
            data={}
            edited_ids={e['uid'] for e in edits}
            for name,label in ((key,'bank'),(alt,'control')):
                state,env=restore(checkpoint['snapshot'],checkpoint['configuration'])
                windows=[]
                for window in (0,1):
                    events=[];acc=Counter();perworker=defaultdict(Counter);bank_events=[]
                    for j in range(10):
                        turns=10*window+j;at=191+turns
                        events.clear()
                        with referee_events(events):step(state,env,logs[name][turns])
                        for k,side,amount,*meta in events:
                            if k=='bank' and amount>0:
                                acc[f's{side}_bank']+=amount
                                bank_events.append([at,side,meta[0],round(amount,2)])
                            if k in ('bank','harvest','loot','enemy_tree','death') and meta and meta[0] in edited_ids:
                                uid=meta[0];perworker[uid][k]+=amount if k!='death' else 1
                            if k=='harvest':acc[f's{side}_harvest']+=amount
                            if k=='death':acc[f's{side}_deaths']+=1
                    t=200+window*10;statecash=[p[0] for p in state[0].observation.players]
                    assert statecash==past[name]['milestones'][str(t)]['cash']
                    uid_deposit=sum(v['bank'] for v in perworker.values());acc['changed_workers_bank']=uid_deposit
                    acc['unchanged_workers_bank']=acc['s0_bank']-uid_deposit
                    acc['changed_workers_harvest']=sum(v['harvest'] for v in perworker.values())
                    acc['changed_workers_loot']=sum(v['loot'] for v in perworker.values())
                    vals={k:round(v,3) for k,v in sorted(acc.items())}
                    windows.append(dict(window_end=t,totals=vals,cash=statecash,bank_events=bank_events))
                data[label]=windows
            deltas=[]
            for w in (0,1):
                a=data['bank'][w]['totals'];b=data['control'][w]['totals']
                combos=set(a)|set(b)
                deltas.append({k:round(a.get(k,0)-b.get(k,0),3) for k in sorted(combos)})
            pairs.append(dict(key=key,root_changed=edits,delta=deltas,paths=data,
                              final_margin_delta=round(past[key]['milestones']['299']['margin']-past[alt]['milestones']['299']['margin'],2)))
        allk=sorted({k for r in pairs for d in r['delta'] for k in d});av=[]
        for w in (0,1):
            av.append({k:round(float(np.mean([r['delta'][w].get(k,0) for r in pairs])),2) for k in allk})
        result[prefix]=dict(pairs=len(pairs),averaged_deltas=av,root_edit_groups=dict(Counter(tuple((e['uid'],e['original'],e['bank']) for e in r['root_changed']).__repr__() for r in pairs)),rows=pairs)
    result['scope']='One post-turn-190 initial state, two families of opponent proxies. Received action logs only; archived cash checked at both horizons. Effects include later decisions by both teams.'
    (OUT/'convoy-branches.json').write_text(json.dumps(result,indent=2)+'\n')
    for k,v in result.items():
        if isinstance(v,dict):print(k,v['root_edit_groups'],v['averaged_deltas'])
if __name__=='__main__':main()
