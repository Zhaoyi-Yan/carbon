"""Exercise joint constraints and replay each saved selected action pair with the vendored referee."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import ast,copy,gzip,json
from math import inf
from statistics import mean,median
from collections import Counter
import numpy as np
from branch_replay import ROOT,OUT,HERE,digest,step
from benchmark import configure_engine,new_game,validate_actions
from close_replay import archive_state,restore,referee_events
from phase_switch import output
from joint_assign import propose


def fixtures(algorithm=propose):
    src=(HERE/'checkpoints/harbormaster.py').read_text()
    f,= [n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='assign_moves']
    ns={'inf':inf};exec(compile(ast.Module([f],type_ignores=[]),'assign_moves','exec'),ns);assign=ns['assign_moves']
    assert 225 in assign([{1:(5,0),225:(-10000,-1)},{1:(6,0),226:(-10000,-1)}])
    payload={'player':0,'players':[[300,{'base':38},{'a':[0,0,'COLLECTOR'],'b':[2,0,'PLANTER'],'other':[100,5,'COLLECTOR']},{}],[100,{'otherbase':217},{},{}]]}
    lw=np.full((225,5),-9.);lw[0,2]=5.;lw[0,0]=4.7;lw[2,4]=10;lw[2,0]=1.;lw[100,0]=9.
    base={'a':'RIGHT','base':'RECCOLLECTOR'}
    target,info,_=algorithm(payload,base,lw,(),assign)
    assert target=={'b':'LEFT','base':'RECCOLLECTOR'} and len(info['edits'])==2 and abs(info['gain']-8.7)<.001
    assert algorithm(payload,base,lw,{'a'},assign)[1] is None
    assert algorithm(payload,base,lw,(),lambda rows:[225+i for i in range(len(rows))])[1] is None
    assert algorithm(payload,dict(base,b='LEFT'),lw,(),assign)[1] is None
    collision=copy.deepcopy(payload);collision['players'][0][1]['base']=1
    collision['players'][0][2].update(a=[0,80,'COLLECTOR'],b=[2,65,'COLLECTOR'])
    pay={'a':'RIGHT','b':'LEFT'}
    same,intent,_=algorithm(collision,pay,lw,(),assign)
    assert same==pay and intent is None
    bad=copy.deepcopy(payload);bad['players'][0][1]['base']=0
    assert algorithm(bad,base,lw,(),assign)[1] is None
    bad=copy.deepcopy(payload);bad['players'][0][1]['bank']=1;bad['players'][0][2]['a'][1]=35
    assert algorithm(bad,base,lw,(),assign)[1] is None
    swap=copy.deepcopy(payload);swap['players'][0][2]['b'][0]=1;lw[1,4]=12;lw[1,0]=1
    act,meta,_=algorithm(swap,{},lw,(),assign);assert act=={'a':'RIGHT','b':'LEFT'} and meta['type']=='yield-to-occupied-destination'
    assert algorithm(payload,base,lw,(),assign,min_gain=20)[1] is None


def verify(c,raw):
    from zerosum_env.envs.carbon import idgen
    fs=json.load(gzip.open(ROOT.parent/c['source']));cfg=fs['configuration'];fs=fs['steps'];new_game(cfg['randomSeed'],300)
    start=dict(seats=[dict(observation=x['observation'],status=x['status'],reward=x['reward'],action=x['action'] or {}) for x in fs[0]],id_counters=dict(idgen._id_dict))
    state,env=restore(start,cfg);r=c['branch'];seat=c['seat']
    for frame in fs[1:r['start']+1]:step(state,env,[s['action'] or {} for s in frame])
    snap=archive_state(state);assert digest(snap)==r['initial_sha256']
    for arm in ('candidate','baseline'):
        state,env=restore(snap,cfg);deps=[0.,0.];record=r[arm]
        for offset,actions in enumerate(raw[arm],1):
            for i,a in enumerate(actions):validate_actions(a,state[0].observation.players[i])
            ev=[]
            with referee_events(ev):step(state,env,actions)
            for name,side,amount,*_ in ev:
                if name in ('bank','base_collision'):deps[side]+=amount
            mark=record['marks'].get(str(offset))
            if mark:assert mark['cash']==[p[0] for p in state[0].observation.players] and mark['deposits']==[round(v,3) for v in deps]
    actions=[raw[n][0] for n in ('baseline','candidate')]
    assert actions[0][1-seat]==actions[1][1-seat]
    changed={u for u in actions[0][seat].keys()|actions[1][seat].keys() if actions[0][seat].get(u)!=actions[1][seat].get(u)}
    assert changed=={e[0] for e in r['raw_offer']['edits']} and len(changed)>=2


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--relay',action='store_true');a=ap.parse_args()
    from joint_relay import propose as relay
    fixtures(relay if a.relay else propose);configure_engine()
    prefix='joint-relay' if a.relay else 'joint-assign'
    d=json.load(open(OUT/(prefix+'.json')));raw=json.loads(gzip.decompress((OUT/(prefix+'-actions.json.gz')).read_bytes()));metrics={}
    for group in ('development','holdout'):
        cs=[r for r in d['cases'] if r['split']==group];assert len(cs)==(12 if group=='development' else 9)
        counts=d['cohorts'][group]['counts'];active=[r for r in cs if r['branch']]
        for r in active:verify(r,raw[r['layout']])
        def rate(which,den,subset):
            return round(100*counts.get(subset+which+'_correct',0)/counts[den],3) if counts.get(den) else None
        accuracy={}
        for key,den,role_prefix in [('all','workers',''),('edited','changed','changed_'),('collector','COLLECTOR_changed','COLLECTOR_'),('planter','PLANTER_changed','PLANTER_')]:
            accuracy[key]={'workers':counts.get(den,0),'old':rate('baseline',den,role_prefix),'new':rate('alternative',den,role_prefix)}
        signed_map_counts=Counter()
        for r in cs:
            s=r['stats'];dlt=s.get('changed_alternative_correct',0)-s.get('changed_baseline_correct',0)
            signed_map_counts['positive' if dlt>0 else ('negative' if dlt<0 else 'equal')]+=1
        metrics[group]=dict(accuracy=accuracy,map_accuracy_signs=dict(signed_map_counts),offer_per_100=round(100*counts.get('offered',0)/counts['turns'],2),paired_maps=len(active),
            timing_us=d['cohorts'][group]['decision_us'],horizons={h:{key:dict(mean=round(mean(r['branch']['delta'][h][key] for r in active),2),median=round(median(r['branch']['delta'][h][key] for r in active),2)) for key in ('cash','deposits')} for h in ('5','10','20')} if active else {})
    output(OUT/(prefix+'-summary.json'),dict(verified_arms=len(raw)*2,groups=metrics));print(metrics)


if __name__=='__main__':main()
