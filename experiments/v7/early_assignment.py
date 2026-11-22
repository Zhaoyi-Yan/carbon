"""Compare which depot starts each profession and whether staggering the two purchases helps."""
import argparse,json,os
from pathlib import Path
from statistics import mean
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[v]='1'
import opening_probe as op
from branch_replay import HERE,OUT
from benchmark import configure_engine


def opening(mode,o,cfg,action,logits,ever_tree,m):
    if mode=='incumbent' or o['step']>1:return action
    side=o['player'];own=o['players'][side];bases=sorted(own[1]);roles=('RECCOLLECTOR','RECPLANTER')
    # Soundings, Pilotage and Fairlead chose this same initial role/seat order on every map.
    first=side if mode!='swap' else 1-side
    assigned={bases[first]:roles[0],bases[1-first]:roles[1]}
    stagger=mode in ('collector_first','planter_first')
    if o['step']==0:
        if stagger:return {uid:act for uid,act in assigned.items() if act==roles[int(mode=='planter_first')]}
        return assigned
    if not stagger:return action
    target=roles[int(mode=='collector_first')]
    out={u:a for u,a in action.items() if u not in own[1]}
    for u,role in assigned.items():
        if role==target and own[0]>=30:out[u]=role
    return out


def test(seeds,modes,mod):
    rows=[]
    for seed in seeds:
        for mode in modes:
            for seat in (0,1):
                game=op.create(seed,mode,seat);op.advance(game,299,mod)
                steps={}
                for tick,values in game['marks'].items():
                    ours,enemy=values[seat],values[1-seat]
                    steps[tick]=dict(cash=round(ours['cash']-enemy['cash'],2))
                rows.append(dict(seed=seed,layout=game['layout'],mode=mode,seat=seat,terminal=steps[str(game['turn'])]['cash'],steps=steps))
        print(seed,len(rows),flush=True)
    return rows


def summary(rows):
    out={};base={(r['seed'],r['seat']):r['terminal'] for r in rows if r['mode']=='usual'}
    for mode in sorted({r['mode'] for r in rows}):
        group=[r for r in rows if r['mode']==mode]
        dif=[r['terminal']-base[r['seed'],r['seat']] for r in group]
        out[mode]=dict(wins=sum(r['terminal']>0 for r in group),mean=round(mean(r['terminal'] for r in group),2),
            paired=round(mean(dif),2),better_seats=sum(d>0 for d in dif),
            seat_gain=[round(mean(r['terminal']-base[r['seed'],r['seat']] for r in group if r['seat']==s),2) for s in (0,1)])
    return out


def main():
    configure_engine();op.variant=opening;mod=op.load_policy();design=json.loads((HERE/'panels.json').read_text())
    dev=design['development'];test_maps=[911020373,422618179,304794743,78324181]
    protected={k for group in ('qualifying','confirmation') for row in design[group] for k in row}
    assert not set(test_maps)&(protected|set(dev))
    raw=test(dev,('usual','swap','collector_first','planter_first'),mod);metrics=summary(raw)
    choices=['swap','collector_first','planter_first']
    chosen=max(choices,key=lambda k:min(metrics[k]['seat_gain']))
    path=OUT/'early-assignment.json';z=dict(dev=metrics,dev_rows=raw,chosen=chosen,holdout_seeds=test_maps,
        method='Both sides use the fixed-turn v6 root and fresh responses. Only the candidate first two depot commands change. Four development maps choose the strongest worst-seat paired terminal gain before opening four untouched maps. Scores are an opening experiment, not release gate results.')
    path.write_text(json.dumps(z,indent=2)+'\n');print(metrics,chosen,flush=True)
    held=test(test_maps,('usual',chosen),mod);z.update(holdout=summary(held),holdout_rows=held)
    path.write_text(json.dumps(z,indent=2)+'\n');print(z['holdout'],flush=True)
if __name__=='__main__':main()
