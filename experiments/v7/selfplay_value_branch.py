"""Measure root-relative ranking on 12 independent archived, multi-policy counterfactual layouts."""
import selfplay_value_data as d
from itertools import combinations
import gzip,hashlib,json
import numpy as np
from close_replay import archive_state,restore

def result(arms,values,goal):
    wins=[];weighted=[];picked=[]
    for candidates in arms:
        group=list(candidates.values())
        if len(group)<2:continue
        best=max(v[goal] for v in group)
        chosen=max(group,key=lambda v:v[values])
        picked.append(best-chosen[goal])
        for a,b in combinations(group,2):
            true=a[goal]-b[goal];pred=a[values]-b[values]
            if abs(true)<=1.:continue
            wins.append(.5 if abs(pred)<1.e-5 else float(pred*true>0));weighted.append(abs(true))
    return dict(distinct_maps=len(picked),eligible_pairs=len(wins),agreement=float(np.mean(wins)) if wins else None,
                weighted_agreement=float(np.average(wins,weights=weighted)) if wins else None,mean_best_regret=round(float(np.mean(picked)),2) if picked else None)

def scan(m,weights,fit,source,case,tapes):
    side=case['side'];init=150;seed=case['seed']
    state,env=d.new_game(seed,300);cfg=env.configuration;tr=d.Features(m,side)
    for t,(cmd,expected) in enumerate(source['actions_and_post_cash'][:init]):
        ob=d.observe(state,side,t);econ,_=m._v5_observe(ob['players'],ob['trees'],t,298);tr.history(ob,cfg,econ,False);tr.mark(ob,cmd[side])
        d.step(state,env,cmd);assert [p[0] for p in state[0].observation.players]==expected
    assert d.board_digest(state)==case['initial_board_sha256']
    snap=archive_state(state);ob=d.observe(state,side,init);econ,_=m._v5_observe(ob['players'],ob['trees'],init,298)
    _,style,_=tr.history(ob,cfg,econ,True);b0={'players':ob['players'],'trees':ob['trees'],'step':init,'carbon':tr.root_stock}
    assert len({json.dumps(v[0][1-side],sort_keys=True) for v in tapes.values()})==1
    bydepth={}
    for name,trace in tapes.items():
        state,env=restore(snap,cfg);cash=[[v[0] for v in state[0].observation.players]]
        pred={};sim=b0
        for tick,action in enumerate(trace,1):
            if tick<=3:
                before=sim['players'];sim=m._turn(sim,cfg,action)
                if tick in (1,3):
                    x=m._v5_pair(sim['players'],sim['step'],sim['carbon'],before)[side]
                    sc=m._scalars(sim['players'],sim['step'],side)
                    h,v,_=d.features(m,weights,[(x,style,sc)])
                    native=m._net()(x[None,:],np.array([style]),sc[None,:])[2][0];assert abs(native-v[0])<.00005
                    sig=hashlib.sha256(x.tobytes()+sc.tobytes()).hexdigest()
                    pred[tick]={'base':float(3000*v[0]),'new':float(3000*(v[0]+h[0]@fit['dw']+fit['db'])),'signature':sig}
            d.step(state,env,action);cash.append([v[0] for v in state[0].observation.players])
        ref=case['arms'][name];assert d.board_digest(state)==ref['terminal_hash'] and [p.reward for p in state]==ref['scores']
        cm=(np.asarray(cash)[:,side]-np.asarray(cash)[:,1-side]);end=len(trace)
        for tick,item in pred.items():
            yy=.28*cm[min(end,tick+20)]+.52*cm[min(end,tick+52)]+.20*cm[min(end,tick+100)]
            item.update(name=name,weighted=float(yy),terminal=float(cm[-1]),target_horizons=[float(cm[min(end,tick+k)]) for k in (20,52,100)])
            bydepth.setdefault(tick,[]).append(item)
    return bydepth


def main():
    d.configure_engine();m,w=d.load();fit=dict(np.load(d.HERE/'selfplay_value_model.npz'));fact=json.loads((d.OUT/'selfplay-value-factual.json').read_text())
    layouts={r['layout'] for r in fact['schedule']};rows={};sets=[]
    names={'v6r-qual-05-estuary.jsonl','v6r-confirm-05-estuary.jsonl'}
    names.update(json.loads((d.OUT/'phase-deficit-schedule.json').read_text())['archived_sources'])
    assert all(n.startswith('v6r-') for n in names)
    for n in sorted(names):
        for line in (d.ROOT/'analysis'/n).read_text().splitlines():
            q=json.loads(line);rows.setdefault(q['seed'],[]).append(q)
    for family in ('phase-switch','phase-deficit'):
        source=json.loads((d.OUT/(family+'.json')).read_text());tapes=json.loads(gzip.decompress((d.OUT/(family+'-actions.json.gz')).read_bytes()))
        for case in source['cases']:
            side=case['side'];key=str(case['seed'])+'/'+str(side)
            assert case['layout'] not in layouts;layouts.add(case['layout'])
            row=next(g for g in rows[case['seed']] if [v['name'] for v in g['players']]==case['historical_names'] and
                    g['actions_and_post_cash'][149][1]==case['initial'])
            data=scan(m,w,fit,row,case,tapes[key])
            sets.append(dict(seed=case['seed'],seat=side,layout=case['layout'],family=family,branches=data))
    report={}
    for depth in (1,3):
        groups=[]
        for case in sets:
            groupset={}
            for item in case['branches'][depth]:groupset.setdefault(item['signature'],[]).append(item)
            groups.append({k:{field:float(np.mean([r[field] for r in vals])) for field in ('base','new','weighted','terminal')} for k,vals in groupset.items()})
        report[depth]={target:{name:result(groups,name,target) for name in ('base','new')} for target in ('weighted','terminal')}
    (d.OUT/'selfplay-value-ranking.json').write_text(json.dumps(dict(summary=report,cases=sets,model_sha256=fact['model_sha256'],
            design='All 12 roots and 36 archived policy continuations from historical phase-switch / phase-deficit are disjoint from the 64 tournament layouts. Reconstruct real prefix. Score the candidate first move, or first three joint moves, from the exact same legal root belief, using the frozen agent one-step simulator. Compare with verified observed 20/52/100 future cash, and separately terminal margin. Equal leaf signatures are merged; future continuations sharing them are averaged. Changes after the first step reflect their own policies. No tuning on this test.'),indent=1)+'\n')
    print('ranking',report,flush=True)

if __name__=='__main__':main()
