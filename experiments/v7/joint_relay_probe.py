"""Evaluate yielding chains in completed root decisions on predecision contested states."""
import os,sys
keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
if __name__=='__main__' and (any(os.environ.get(k)!='1' for k in keys) or os.environ.get('PYTHONHASHSEED')!='0'):
    os.environ.update(dict.fromkeys(keys,'1'),PYTHONHASHSEED='0');os.execv(sys.executable,[sys.executable,*sys.argv])
import hashlib,json,gzip,time
from collections import Counter
from statistics import mean
import numpy as np
import joint_assign_probe as old
from joint_relay import propose
from phase_switch import single_thread,preserved_decision,output
from mine_bank_probe import executed,FROZEN
from benchmark import configure_engine
from branch_replay import OUT,ROOT,HERE


def main():
    runtime=single_thread();configure_engine();os.environ['CARBON_DEBUG']='1';old.propose=propose
    paths=[HERE/'checkpoints/harbormaster.py',ROOT/'experiments/v6r/checkpoints/estuary.py']
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==FROZEN[n] for n,p in zip(FROZEN,paths))
    models=[old.model(p,'joint_relay_'+n,True) for n,p in zip(FROZEN,paths)]
    records=[];raw={};timing={'development':[],'holdout':[]};summary={}
    data=dict(frozen=FROZEN,runtime=runtime,rule_sha256=hashlib.sha256((HERE/'joint_relay.py').read_bytes()).hexdigest(),
     protocol='Fixed before reading primary held-out results. Grow genuine greedy preferred-destination conflicts over two highest legal cells within 2 raw logits, permit only connected changed workers trading original destination. On such turns, finish four-candidate/depth-two fixed root search; use same logits against its completed legal joint action. Min logit gain .25 (odds ratio 2.30 at root temperature .3). Expert accuracy compares completed root (costly on contested frames only) to constrained joint offer. Up to 3 prospective hash-ranked windows, at most one paired live branch per map. No later expert results for selection.',cases=records,cohorts=summary)
    def save():
        for g in timing:
            cs=[c for c in records if c['split']==g];tot=sum((Counter(c['stats']) for c in cs),Counter());tests=[c['branch'] for c in cs if c['branch']]
            summary[g]=dict(maps=len(cs),counts=tot,decision_us={str(q):round(float(np.percentile(timing[g],q)),2) for q in (50,95,99)} if timing[g] else {},
                 paired_maps=len(tests),mean_delta={h:{k:round(mean(r['delta'][h][k] for r in tests),2) for k in ('cash','deposits')} for h in ('5','10','20')} if tests else {},
                 positive20=sum(r['delta']['20']['cash']>0 for r in tests),negative20=sum(r['delta']['20']['cash']<0 for r in tests))
        output(OUT/'joint-relay.json',data);output(OUT/'joint-relay-actions.json.gz',raw,True)
    for split,rec,path in old.selections():
        game=json.load(gzip.open(path));cfg=game['configuration'];fs=game['steps'];side=rec['seat'];k=rec['layout']
        salt=int(hashlib.sha256(('joint-assign-salt/'+k).encode()).hexdigest()[:8],16)%100000;p=models[0]
        p._ST={};p._FC_HISTORY={};stats=Counter();valid=[];hist=Counter()
        vis={n:v for n,v in cfg.items() if n!='randomSeed'}
        def tap(o,c,stock,base,lw,lb,prev,t0,home=()):
            fixed=set(home)|set(p._r6_early_locks(o,base,lw));start=time.perf_counter_ns()
            _,_,conf=propose(o,base,lw,fixed,p.assign_moves)
            timing[split].append((time.perf_counter_ns()-start)/1000.)
            p.record=(lw,fixed,conf)
            if conf:p._reply_choice=p.reply_full;return p.search_full(o,c,stock,base,lw,lb,prev,t0,home)
            return base
        for t in range(len(fs)-1):
            pub=fs[t][0]['observation'];ob=dict(player=side,step=t,remainingOverageTime=5,players=pub['players'],trees=pub['trees'],carbon=fs[t][side]['observation']['carbon'])
            p.record=None;old.budget(p,False);p._search=tap;normal=preserved_decision(p,ob,vis,salt);actual=fs[t+1][side]['action'] or {}
            executed(p,ob,normal,actual)
            if not p.record or t<5:continue
            lw,fixed,conf=p.record;stats['turns']+=1;stats['contested_turns']+=bool(conf);alt=normal;meta=None
            if conf:alt,meta,_=propose(ob,normal,lw,fixed,p.assign_moves)
            for u in pub['players'][side][2]:
                stats['workers']+=1;stats['baseline_correct']+=normal.get(u)==actual.get(u);stats['alternative_correct']+=alt.get(u)==actual.get(u)
            if not meta:continue
            stats['offered']+=1;hist[len(meta['edits'])]+=1
            for u,b,a,role in meta['edits']:
                stats['changed']+=1;stats['changed_baseline_correct']+=b==actual.get(u);stats['changed_alternative_correct']+=a==actual.get(u)
                stats[role+'_changed']+=1;stats[role+'_baseline_correct']+=b==actual.get(u);stats[role+'_alternative_correct']+=a==actual.get(u)
            if 18<=t<len(fs)-20:valid.append((hashlib.sha256(f'joint-relay-window/{k}/{t}/{side}'.encode()).digest(),t))
        case=dict(split=split,seed=rec['seed'],layout=k,teacher=rec['teacher'],seat=side,match=rec['match'],game=rec['game'],source=str(path.relative_to(ROOT.parent)),stats=stats,edit_histogram=hist,attempted=[],branch=None)
        for _,t in sorted(valid)[:3]:
            case['attempted'].append(t);pair=old.compare(cfg,fs,t,side,k,salt,models)
            if pair:
                raw[k]={a:pair[a].pop('actions') for a in ('candidate','baseline')};case['branch']=pair;break
        records.append(case);save();print(split,len(records),stats['offered'],case['branch']['delta']['20'] if case['branch'] else None,flush=True)


if __name__=='__main__':main()
