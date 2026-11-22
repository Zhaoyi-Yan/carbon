"""Verify both saved arms with the official referee and summarize held-out outcomes by map."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import gzip
import json
import random
from statistics import mean,median
from collections import Counter
from branch_replay import ROOT,OUT,digest,step
from preempt_tree_probe import parse_public
from close_replay import archive_state,restore,referee_events
from benchmark import new_game,configure_engine,validate_actions
from phase_switch import output


def verify(c,traces,index):
    from zerosum_env.envs.carbon import idgen
    cfg,fs,k=parse_public(index[c['layout']]);assert cfg['randomSeed']==c['map_seed'];new_game(cfg['randomSeed'],300)
    start=dict(seats=[dict(observation=x['observation'],status=x['status'],reward=x['reward'],action=x['action'] or {}) for x in fs[0]],id_counters=dict(idgen._id_dict))
    state,env=restore(start,cfg)
    for f in fs[1:c['start']+1]:
        step(state,env,[p['action'] or {} for p in f])
    snap=archive_state(state);assert digest(snap)==c['snapshot_sha256']
    extra={}
    for arm in ('candidate','baseline'):
        cash_flow=[Counter(),Counter()];extra[arm]={}
        state,env=restore(snap,cfg);uid=c[arm]['initial']['uid'];side=c['side'];bank=[0.,0.];collision=[0.,0.];collected=[0.,0.];amounts=[]
        for off,actions in enumerate(traces[k][arm],1):
            for s,a in enumerate(actions):validate_actions(a,state[0].observation.players[s])
            events=[]
            with referee_events(events):step(state,env,actions)
            for name,s,v,*detail in events:
                if name in ('hire_collector','hire_planter'):cash_flow[s]['hire_payment']+=v
                elif name in ('plant','seize'):cash_flow[s]['tree_payment']+=v
                elif name=='tree_collision':cash_flow[s]['tree_collision']+=v
                if name=='bank':
                    bank[s]+=v
                    if s==side and detail[0]==uid:amounts.append([off,round(v,3)])
                elif name=='base_collision':collision[s]+=v
                elif name=='harvest' and s==side and detail[0]==uid:collected[s]+=v
            mark=c[arm]['marks'].get(str(off))
            if mark:
                extra[arm][str(off)]={k:cash_flow[side][k]-cash_flow[1-side][k] for k in ('hire_payment','tree_payment','tree_collision')}
                for s,p in enumerate(state[0].observation.players):
                    assert [mark[s]['cash'],mark[s]['bank_only'],mark[s]['deposits'],mark[s]['tracked_mined']]==[p[0],round(bank[s],3),round(bank[s]+collision[s],3),round(collected[s],3)]
        assert amounts==c[arm]['deliveries']
    for h in ('3','5','10','20'):
        a,b=(extra[r][str(min(int(h),c[r]['end']))] for r in ('candidate','baseline'))
        c['delta'][h].update({k:a[k]-b[k] for k in a})


def describe(rows):
    if not rows:return {'n':0}
    n=len(rows);rand=random.Random(190927);idx=[[rand.randrange(n) for _ in range(n)] for i in range(5000)]
    def stats(vals):
        boots=sorted(sum(vals[j] for j in ix)/n for ix in idx)
        return dict(mean=round(mean(vals),2),median=round(median(vals),2),positive=sum(v>0 for v in vals),negative=sum(v<0 for v in vals),
                    map_bootstrap_95_interval=[round(boots[125],2),round(boots[4874],2)])
    horizons={}
    for h in (3,5,10,20):
        out=horizons[str(h)]={k:stats([r['delta'][str(h)][k] for r in rows]) for k in ('cash','deposits','tracked_mined','tree_income','hire_payment','tree_payment','tree_collision')}
        tracked={}
        for arm in ('baseline','candidate'):
            paid=[sum(v for off,v in r[arm]['deliveries'] if off<=h) for r in rows]
            tracked[arm]=dict(banked=sum(v>0 for v in paid),mean=round(mean(paid),2))
        out['target_collector']=tracked
    return dict(n=n,horizons=horizons,controlled_seats=[sum(r['side']==s for r in rows) for s in (0,1)],
        first_mission_outcomes=[dict(layout=r['layout'],start=r['start'],side=r['side'],distance=r['candidate']['initial']['dist'],cash=r['candidate']['initial']['cash'],
              route=r['candidate']['edits'],payouts_first3=[v for t,v in r['candidate']['deliveries'] if t<=3]) for r in rows])


def main():
    configure_engine();d=json.loads((OUT/'mine-bank.json').read_text());traces=json.loads(gzip.decompress((OUT/'mine-bank-actions.json.gz').read_bytes()))
    assert {'development','holdout'}==set(d['panels']) and not set(d['schedule']['development'])&set(d['schedule']['holdout'])
    games=json.loads((ROOT.parent/'v7-data/receipts-cache/index.json').read_text())['games'];index={r['layout']:r for r in games}
    reports={}
    for name,p in d['panels'].items():
        rows=p['results']
        assert len({r['layout'] for r in rows})==len(rows) and all(r['layout'] in d['schedule'][name] for r in rows)
        for r in rows:verify(r,traces,index)
        reports[name]=dict(searched_maps=len(p['scanned']),historically_eligible=sum(len(row)>2 for row in p['scanned']),**describe(rows))
    output(OUT/'mine-bank-summary.json',dict(verified_paired_windows=sum(p['n'] for p in reports.values()),cohorts=reports))
    print({k:v['n'] for k,v in reports.items()})


if __name__=='__main__':main()
