"""Measure a local Hungarian offer against disjoint public experts and responding frozen policies."""
import os,sys
keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
if __name__=='__main__' and (any(os.environ.get(k)!='1' for k in keys) or os.environ.get('PYTHONHASHSEED')!='0'):
    os.environ.update(dict.fromkeys(keys,'1'),PYTHONHASHSEED='0');os.execv(sys.executable,[sys.executable,*sys.argv])
import copy,gzip,hashlib,json,time
from statistics import mean
from collections import Counter
import numpy as np
from branch_replay import ROOT,HERE,OUT,digest,step
from close_replay import restore,referee_events
from phase_switch import output,single_thread,preserved_decision
from mine_bank_probe import FROZEN,original,executed
from preempt_tree_probe import model,budget
from benchmark import configure_engine,observe,validate_actions
from joint_assign import propose


def capture(p,full):
    budget(p,full);base_fn=p._search
    def called(obs,cfg,stock,base,lw,lb,prev,t0,home=()):
        p.record=(dict(base),lw,set(home)|set(p._r6_early_locks(obs,base,lw)))
        return base_fn(obs,cfg,stock,base,lw,lb,prev,t0,home)
    p.record=None;p._search=called


def selections():
    archive=ROOT.parent/'v6r-data/artemis';fresh=ROOT.parent/'v7-data/surrogate-holdout'
    files={}
    for root,rows in ((archive,json.load(open(archive/'index.json'))),(fresh,json.load(open(fresh/'index.json'))['rows'])):
        for r in rows:files[(r['id'],r['game'])]=root/r['file']
    audit=json.load(open(OUT/'surrogate-validation.json'))['games']
    training={i.split('-')[0] for i in json.load(open(ROOT/'experiments/v6r/elite-validation.json'))['training']}
    economic={r['seed'] for r in json.load(open(ROOT.parent/'v7-data/receipts-cache/index.json'))['games']}
    assert len({r['layout'] for r in audit})==len(audit)
    for r in audit:
        assert r['match'] not in training
        if r['cohort']=='post_training':assert r['seed'] not in economic
        yield ('holdout' if r['cohort']=='post_training' else 'development'),r,files[(r['match'],r['game'])]


def teach(p,cfg,fs,side,k,salt):
    p._ST={};p._FC_HISTORY={};capture(p,False)
    stats=Counter();us=[];hist=Counter();candidates=[]
    vis={i:v for i,v in cfg.items() if i!='randomSeed'}
    for t in range(len(fs)-1):
        report=fs[t][side]['observation'];public=fs[t][0]['observation']
        ob=dict(player=side,step=t,remainingOverageTime=5,carbon=report['carbon'],players=public['players'],trees=public['trees'])
        p.record=None;normal=preserved_decision(p,ob,vis,salt);actual=fs[t+1][side]['action'] or {};executed(p,ob,normal,actual)
        if t<5 or p.record is None:continue
        raw,lw,locks=p.record
        start=time.perf_counter_ns();alt,offer,conflicts=propose(ob,raw,lw,locks,p.assign_moves);us.append((time.perf_counter_ns()-start)/1000.)
        stats['turns']+=1;stats['contested_turns']+=bool(conflicts);stats['offered']+=bool(offer)
        for uid in ob['players'][side][2]:
            stats['workers']+=1;stats['baseline_correct']+=raw.get(uid)==actual.get(uid);stats['alternative_correct']+=alt.get(uid)==actual.get(uid)
        if not offer:continue
        hist[len(offer['edits'])]+=1
        for u,b,a,role in offer['edits']:
            stats['changed']+=1;stats['changed_baseline_correct']+=b==actual.get(u);stats['changed_alternative_correct']+=a==actual.get(u)
            stats[role+'_changed']+=1;stats[role+'_baseline_correct']+=b==actual.get(u);stats[role+'_alternative_correct']+=a==actual.get(u)
        if 18<=t<len(fs)-20:
            rank=hashlib.sha256(f'joint-window/{k}/{t}/{side}'.encode()).digest()
            candidates.append((rank,t,offer))
    return stats,us,hist,sorted(candidates)


def branch(cfg,vis,snap,mem,mods,t,s,salt,treatment):
    state,env=restore(snap,cfg)
    for p,(st,fc) in zip(mods,copy.deepcopy(mem)):p._ST,p._FC_HISTORY=st,fc
    marks={};history=[];bank=[0.,0.]
    for off in range(20):
        actions=[]
        for side,p in enumerate(mods):
            o=observe(state,side,t+off)
            if off==0 and s==side:capture(p,True)
            else:budget(p,off==0)
            normal=preserved_decision(p,o,vis,salt);a=normal
            if off==0 and side==s:
                _,lw,locked=p.record
                alt,meta,_=propose(o,normal,lw,locked,p.assign_moves)
                if not meta:return None
                if treatment:a=alt
            validate_actions(a,o['players'][side]);executed(p,o,normal,a);actions.append(a)
        history.append(actions)
        events=[]
        with referee_events(events):step(state,env,actions)
        for name,side,value,*_ in events:
            if name in ('bank','base_collision'):bank[side]+=value
        if (off+1) in (1,5,10,20) or any(x.status!='ACTIVE' for x in state):
            marks[str(off+1)]={'cash':[v[0] for v in state[0].observation.players],'deposits':[round(v,3) for v in bank]}
        if any(x.status!='ACTIVE' for x in state):break
    return dict(meta=meta,actions=history,marks=marks,end=off+1)


def compare(cfg,fs,t,s,k,salt,models):
    order=models if s==0 else list(reversed(models))
    snap,mem,vis=original(cfg,fs,t,s,order,salt)
    b=branch(cfg,vis,snap,mem,order,t,s,salt,False)
    if not b:return None
    a=branch(cfg,vis,snap,mem,order,t,s,salt,True)
    assert a['meta']==b['meta'] and a['actions'][0][1-s]==b['actions'][0][1-s]
    actual_changes={u for u in a['actions'][0][s].keys()|b['actions'][0][s].keys() if a['actions'][0][s].get(u)!=b['actions'][0][s].get(u)}
    assert actual_changes=={e[0] for e in a['meta']['edits']}
    delta={}
    for h in (1,5,10,20):
        aa,bb=(v['marks'][str(min(v['end'],h))] for v in (a,b))
        delta[str(h)]={q:round((aa[q][s]-aa[q][1-s])-(bb[q][s]-bb[q][1-s]),3) for q in aa}
    return dict(start=t,side=s,seed=cfg['randomSeed'],layout=k,initial_sha256=digest(snap),salt=salt,raw_offer=a['meta'],baseline=b,candidate=a,delta=delta)


def main():
    runtime=single_thread();configure_engine();os.environ['CARBON_DEBUG']='1'
    paths=[HERE/'checkpoints/harbormaster.py',ROOT/'experiments/v6r/checkpoints/estuary.py']
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==FROZEN[n] for n,p in zip(FROZEN,paths))
    models=[model(p,'joint_assign_'+n,True) for n,p in zip(FROZEN,paths)]
    from torch import get_num_threads,get_num_interop_threads
    assert get_num_threads()==get_num_interop_threads()==1
    cases=[];raw={};snap={};times={'development':[],'holdout':[]}
    data=dict(runtime=runtime,frozen=FROZEN,rule_sha256=hashlib.sha256((HERE/'joint_assign.py').read_bytes()).hexdigest(),
        tested_gain=.75,development='12 historically untouched-by-gradient checkpoint validation boards; some used for epoch selection.',
        holdout='9 post-training Artemis layouts; no matching training game ID in elite-policy list or map seed in 862-game economic manifest.',
        protocol='All historical expert-seat observations from step 5 onward; root raw greedy moves compared to joint offer. Lock go_home, early locks, existing depot payments, all shared-destination workers, occupied hires and other workers. A branch requires a live finished root to emit the proposal, with up to 3 hash-ordered predecision opportunities per layout, then at most one accepted pair per map. Same full root (width4 depth2 replies1 tail3) at fork, both policies recompute cheap roots for all follow-up turns. No repeated intervention.',
        cases=cases,cohorts=snap)
    def save():
        for g in times:
            chosen=[c for c in cases if c['split']==g];counts=sum((Counter(c['stats']) for c in chosen),Counter())
            tests=[c['branch'] for c in chosen if c['branch'] is not None]
            snap[g]=dict(maps=len(chosen),counts=counts,decision_us={str(p):round(float(np.percentile(times[g],p)),2) for p in (50,95,99)} if times[g] else {},
                         paired_maps=len(tests),mean_delta={str(h):{k:round(mean(b['delta'][str(h)][k] for b in tests),2) for k in ('cash','deposits')} for h in (1,5,10,20)} if tests else {},
                         positive20=sum(b['delta']['20']['cash']>0 for b in tests),negative20=sum(b['delta']['20']['cash']<0 for b in tests))
        output(OUT/'joint-assign.json',data);output(OUT/'joint-assign-actions.json.gz',raw,True)
    for split,r,path in selections():
        game=json.load(gzip.open(path));cfg=game['configuration'];fs=game['steps'];s=r['seat'];k=r['layout']
        assert cfg['randomSeed']==r['seed'];salt=int(hashlib.sha256(('joint-assign-salt/'+k).encode()).hexdigest()[:8],16)%100000
        stats,latencies,hist,offers=teach(models[0],cfg,fs,s,k,salt);times[split].extend(latencies)
        case=dict(split=split,seed=r['seed'],layout=k,teacher=r['teacher'],seat=s,match=r['match'],game=r['game'],
                 source=str(path.relative_to(ROOT.parent)),stats=stats,edit_histogram=hist,attempted=[],branch=None)
        for _,t,o in offers[:3]:
            case['attempted'].append(t);pair=compare(cfg,fs,t,s,k,salt,models)
            if pair:
                raw[k]={a:pair[a].pop('actions') for a in ('candidate','baseline')};case['branch']=pair;break
        cases.append(case);save();print(split,len(cases),stats['offered'],stats['changed_baseline_correct'],stats['changed_alternative_correct'],case['branch']['delta']['20'] if case['branch'] else None,flush=True)


if __name__=='__main__':main()
