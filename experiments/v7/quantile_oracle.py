"""Audit the true frozen sampler before search, separately from predictions with the rival's fog."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from distill_data import load,load_manifest,configure_engine,public,local
from receipt_data import ACT
from quantile_fit import ROOT

def main(data):
    configure_engine()
    rows,fold,_=load_manifest(data/'distill-public')
    sample=[]
    for src in ('local','public'):
        used=set()
        for r in sorted(rows,key=lambda a:(a['group'],a['id'])):
            if r['source']==src and fold[r['group']]=='test' and r['group'] not in used:
                used.add(r['group']);sample.append(r)
                if len(used)>=4:break
    m=load(ROOT/'yan-v6.py');assert m._SEARCH_OK and m._SEED==37 and m._ROOT_TEMP==.30
    results=defaultdict(list);records=[];updates=0
    for row in sample:
        m._ST.clear();side=row['seat'];original={}
        def stub(obs,cfg,est,acts,lw,lb,prev,t0,home=()):
            original.update(action=dict(acts),workers=lw,bases=lb)
            return acts
        m._search=stub;m._reply_choice=lambda obs,cfg,est,acts,*rest:acts
        frames=local(row['payload'],side) if row['source']=='local' else public(row['payload'],side)
        tape=np.random.RandomState(37);placebo=np.random.RandomState(36)
        for t,o,_,actions,cfg in frames:
            if t>=64:break
            original.clear();m.LAST_TURN=cfg['episodeSteps']-2
            m._act(o,cfg,None);p=o['players'][side]
            assert 'action' in original
            n=len(p[1])+len(p[2]);u=tape.random_sample(n);v=placebo.random_sample(n)
            a=tape.get_state();b=m._ST['rng'].get_state()
            assert a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
            updates+=1
            for j,(uid,(pos,cargo,k)) in enumerate(p[2].items(),start=len(p[1])):
                prob=m._probs(original['workers'][pos],m._ROOT_TEMP).astype(float)
                cdf=np.cumsum(prob);cdf/=cdf[-1]
                draw=int(np.searchsorted(cdf,u[j],side='right'))
                wrong=int(np.searchsorted(cdf,v[j],side='right'))
                initial=ACT[original['action'].get(uid) or 'NONE']
                final=ACT[actions[side].get(uid) or 'NONE']
                greedy=int(np.argmax(prob));score=[draw==final,wrong==final,greedy==final,draw==initial,initial==final,draw!=greedy]
                for typ,use in [('all',True),('loaded',k=='COLLECTOR' and cargo>=65),('non_greedy',final!=greedy),('loaded_non_greedy',k=='COLLECTOR' and cargo>=65 and final!=greedy)]:
                    if use:
                        for source in ['all',row['source']]:results[source,typ].append(score)
            m._ST['tr'].set_actions(o['players'],actions[side])
        records.append(dict(id=row['id'],source=row['source'],layout=row['group'],seed=row['seed'],side=side))
    out={}
    fields='seed37_final seed36_final greedy_final seed37_joint joint_final raw_draw_differs_greedy'.split()
    for (src,typ),vals in results.items():
        out.setdefault(src,{})[typ]=dict(zip(fields,np.mean(vals,0).round(5).tolist()),units=len(vals))
    report=dict(ledger='All root MT19937 position, Gaussian cache and 624 internal words exactly match after every audited turn. Each base and each worker consumes one scalar in insertion order, including unaffordable bases and forced returns.',rng_checks=updates,
         method='Target-side recorded fog is only used for this upper-bound diagnosis, never for deployed fitting. Frozen agent uses true own root logits including response and cargo residual. Turn-by-turn search and post-search picker bypassed, own action history forced to recorded moves. joint_final disagreement includes actual search, postprocessing, and potential hosting/runtime differences. Public v6 identity is its declared frozen version.',
         window=[0,64],sources=records,results=out)
    (ROOT/'analysis/v7/quantile-oracle.json').write_text(json.dumps(report,indent=2)+'\n')
    print(updates,out['all'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);main(p.parse_args().data)
