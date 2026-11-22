"""Paired seeded rescue from two verified public states, against a reactive Master7sE behavior clone."""
from collections import Counter
import argparse,copy,gzip,hashlib,json,os,time
from pathlib import Path
from types import SimpleNamespace
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from branch_replay import ROOT,OUT,HERE,board_digest,step
from benchmark import limit_math_threads,configure_engine,load_strategy,observe,validate_actions
from close_replay import archive_state,restore
from master_data import policy
import master as old
import master_worker as direction
from receipt_data import ACT,LAND
from starvation_replay import read_replay

MATCH='4063a34afedb6d4767a9118e1dee9e5d'
SEEDS=(18011,38039,62047)
START_END=((164,299),(89,124))
FROZEN={'v6':'e5c328aa5d18ced597791e7b839088f1a37d18e4e8dc83dd3d54b13d66e877b0','harbormaster':'d0ba5f631ceeb9ba5670a0013422caca414e407252e6907b90ebe9b33700be87'}


def shot(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ours(name,path,expected_sha=None):
    if shot(path)!=(expected_sha or FROZEN[name]):raise ValueError('Candidate bytes changed: '+name)
    src=path.read_text()
    start=src.index('    if _S_COST[0] > 0.0:',src.index('def _search('));end=src.index('    stages =',start)
    src=src[:start]+'    most,D2,M=4,min(D2,2),1\n'+src[end:]
    for a,b in [('+ 17 * me) % (2 ** 31)','+ 17 * me + _BRANCH_SEED) % (2 ** 31)'),
                ('for tick in range(1,15):','for tick in range(1,4):'),
                ('_torch.set_num_threads(2)','_torch.set_num_threads(1)'),
                ('return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT','return False')]:
        if src.count(a)!=1:raise ValueError((name,a,src.count(a)))
        src=src.replace(a,b)
    m=load_strategy('master_e_paired_'+name,path,src)
    m._BRANCH_SEED=0;m._S_MMAX=1;m._time=SimpleNamespace(perf_counter=lambda:0.)
    assert m._SEARCH_OK and m._RX and m._torch.get_num_threads()==1
    return m


class Clone:
    def __init__(self):
        self.m=m=policy(HERE/'checkpoints/fairlead.py')
        for key in ('NPL','N','_V5_REACH','_V5_GEOMETRY','_V5_ROTATIONS'):setattr(direction,key,getattr(m,key))
        self.w=dict(np.load(HERE/'master-worker-model.npz'))
        self.b=dict(np.load(HERE/'master-model.npz'))
        self.reset()

    def reset(self):
        self.m._ST={};self.previous=None;self.tr=None

    def predict(self,obs,cfg):
        m=self.m;s=obs['player'];t=obs['step'];pl=obs['players'];mine=pl[s]
        if self.tr is None:self.tr=m.BeliefTracker(m._Bel,s)
        stock,known=self.tr.step(pl,t,obs['carbon'])
        stock,known=m._sense_tree_carbon(self.tr.bel,obs,cfg,self.previous,stock,known)
        x=m._quant(m.encode(pl,t,s,stock,known,self.previous))
        current=m._v5_root(obs,stock,x,self.previous)
        own=m._MS_CAPTURE['own'];latent=m._MS_CAPTURE['h']
        ids,roles,raw,local,context,_=direction._mw_inputs(obs,x,self.previous,latent)
        workers=current[0].copy();bases=current[1].copy()
        if ids:
            cells=[mine[2][u][0] for u in ids];style=np.broadcast_to(np.array([0.,0.,1.],dtype='f'),(len(ids),3))
            workers[cells]=direction._mw_logits(raw,local,context,roles,workers[cells],style,self.w)
        bp=np.asarray(list(mine[1].values()),dtype=int)
        h=sum(latent[j,cells[bp]] for j,(cells,_,_,_) in enumerate(m._V5_ROTATIONS))*.25
        _,bx=old._ms_base_rows(pl,s,t,x,own,current,h)
        specific=self.b['s2'][2]
        bases[bp]=bx@self.b['w2']+bx[:,:,:len(specific)]@specific
        return workers,bases

    def choose(self,obs,heads,seed):
        t=obs['step'];s=obs['player']
        return self.m._sample(obs['players'],s,*heads,np.random.RandomState(seed+1000003*(t+1)+s),t)

    def finish(self,obs,action):
        self.tr.set_actions(obs['players'],action)
        self.previous=obs['players'];self.m._ST['prev']=self.previous

    def snap(self):
        return copy.deepcopy((self.m._ST,self.tr,self.previous))

    def restore(self,state):
        self.m._ST,self.tr,self.previous=copy.deepcopy(state)


def readings(pl,s):
    cash,_,wu,trees=pl[s]
    return dict(cash=round(cash,2),workers=len(wu),collectors=sum(w[2]=='COLLECTOR' for w in wu.values()),
                planters=sum(w[2]=='PLANTER' for w in wu.values()),trees=len(trees),cargo=round(sum(w[1] for w in wu.values()),3))


def deposits(pl,actions):
    held={q:j for j,p in enumerate(pl) for q in p[1].values()};events=[]
    for s,p in enumerate(pl):
        for uid,(q,load,kind) in p[2].items():
            if load<=0:continue
            final=LAND[q][ACT[actions[s].get(uid) or 'NONE']]
            if final in held:events.append((held[final],s,uid,load))
    return events


def history(data,agents,clone):
    from zerosum_env.envs.carbon import idgen
    cfg=data['configuration'];rules={k:v for k,v in cfg.items() if k!='randomSeed'}
    state,env=restore(data['initial'],cfg);snaps={}
    normal={}
    for name,m in agents.items():
        normal[name]=m._search,m._reply_choice
        m._search=lambda obs,cfg,est,a,*args:a
        m._reply_choice=lambda obs,cfg,est,a,*args:a
    starts={a for a,_ in START_END}
    # Neither the function running a fork nor the objects saved below receive future replay actions.
    for t in range(max(starts)+1):
        if board_digest(state)!=data['expected'][t]:raise ValueError(('prefix differs',t))
        if t in starts:
            snaps[t]=(archive_state(state),clone.snap(),{n:copy.deepcopy((m._ST,m._FC_HISTORY)) for n,m in agents.items()})
        if t==max(starts):break
        v=[observe(state,s,t) for s in (0,1)];cmd=data['actions'][t]
        counters=dict(idgen._id_dict)
        clone.predict(v[0],rules);clone.finish(v[0],cmd[0])
        for m in agents.values():
            m.agent(v[1],rules);m._ST['tr'].set_actions(v[1]['players'],cmd[1])
        idgen._id_dict.clear();idgen._id_dict.update(counters)
        step(state,env,cmd)
    for name,m in agents.items():m._search,m._reply_choice=normal[name]
    return snaps


def branch(cfg,snap,clone,m,name,start,end,seed):
    from zerosum_env.envs.carbon import idgen
    state,env=restore(snap[0],cfg);clone.restore(snap[1]);m._ST,m._FC_HISTORY=copy.deepcopy(snap[2][name]);m._BRANCH_SEED=seed
    rules={k:v for k,v in cfg.items() if k!='randomSeed'}
    got=np.zeros((2,2));first={};ledger=[];commands=hashlib.sha256();milestones={}
    for t in range(start,end):
        obs=[observe(state,s,t) for s in (0,1)]
        if any('full_carbon' in x for x in obs):raise ValueError('Private map in actor input')
        counters=dict(idgen._id_dict)
        enemy=clone.choose(obs[0],clone.predict(obs[0],rules),seed);clone.finish(obs[0],enemy)
        m._ST['rng']=np.random.RandomState(seed+1000003*(t+1)+1)
        ours=m.agent(obs[1],rules)
        actions=[enemy,ours];pl=obs[0]['players']
        for a,p in zip(actions,pl):validate_actions(a,p)
        idgen._id_dict.clear();idgen._id_dict.update(counters)
        for recipient,source,uid,load in deposits(pl,actions):
            got[recipient,source]+=load
            key=('own' if recipient==source else 'gift')+'_seat'+str(recipient)
            first.setdefault(key,dict(t=t+1,worker=uid,amount=round(load,3)))
            ledger.append([t+1,recipient,source,uid,round(load,3)])
        commands.update(json.dumps(actions,sort_keys=True,separators=(',',':')).encode()+b'\n')
        step(state,env,actions)
        if t+1 in (start+10,start+25,end) or any(p.status=='DONE' for p in state):
            pl=state[0].observation.players
            milestones[t+1]=dict(seats=[readings(pl,s) for s in (0,1)],
                                gross_by_recipient_source=got.round(3).tolist(),margin_QQ=round(pl[1][0]-pl[0][0],2))
        if any(p.status=='DONE' for p in state):break
    final=milestones[t+1]
    return dict(arm=name,start=start,through=t+1,rng_seed=seed,seats=['Master7sE-BC',name],opening=[readings(snap[0]['seats'][0]['observation']['players'],s) for s in (0,1)],
                first=first,milestones=milestones,final_cash_margin=final['margin_QQ'],rewards=[p.reward for p in state],statuses=[p.status for p in state],
                action_sha256=commands.hexdigest(),deposits=ledger)


def validate_clone(clone,cache):
    rows=[r for r in json.loads((cache/'index.json').read_text()) if r['fold']=='test' and r['teacher']=='Master7sE']
    result=[]
    for r in rows:
        game=read_replay(Path(r['file']));f=game['steps'];cfg=game['configuration'];s=r['side'];clone.reset()
        sums={'all':Counter(),'164+':Counter()}
        for t,frame in enumerate(f[:-1]):
            common=frame[0]['observation'];pl=common['players'];p=pl[s]
            ob=dict(players=pl,player=s,step=t,trees=common['trees'],carbon=frame[s]['observation']['carbon'])
            heads=clone.predict(ob,cfg)
            # Next-frame official action is read only after probabilities were produced.
            gold=f[t+1][s]['action'] or {}
            for seed in SEEDS:
                a=clone.choose(ob,heads,seed);c=Counter()
                for uid,(pos,load,role) in p[2].items():
                    exact=a.get(uid)==gold.get(uid);c['workers']+=1;c['worker_exact']+=exact
                    if role=='COLLECTOR' and load>=65 and min(clone.m.DIST[pos][q] for q in p[1].values())<=4:
                        c['heavy']+=1;c['heavy_exact']+=exact
                    if load:
                        ga=LAND[pos][ACT[gold.get(uid) or 'NONE']] in p[1].values();pa=LAND[pos][ACT[a.get(uid) or 'NONE']] in p[1].values()
                        c['gold_money']+=load*ga;c['pred_money']+=load*pa
                    if load<=15 and role=='COLLECTOR' and min(clone.m.DIST[pos][q] for q in pl[1-s][1].values())<=4:
                        c['pressure']+=1;c['pressure_exact']+=exact
                if p[0]>=30 and len(p[2])<10:
                    for bid in p[1]:
                        c['eligible_bases']+=1;c['hire_exact']+=a.get(bid)==gold.get(bid)
                        c['actual_hire']+=bool(gold.get(bid));c['pred_hire']+=bool(a.get(bid))
                sums['all'].update(c)
                if t>=164:sums['164+'].update(c)
            clone.finish(ob,gold)
        result.append(dict(match=r['match'],seed=r['seed'],seat=s,counts={k:dict(v) for k,v in sums.items()}))
        print('validated',r['match'],flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',required=True,type=Path);p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--out',default=OUT/'master-e-rescue.json',type=Path);p.add_argument('--progress',type=Path);a=p.parse_args()
    limit_math_threads();configure_engine();os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    agents={'v6':load_ours('v6',ROOT/'yan-v6.py'),'harbormaster':load_ours('harbormaster',a.candidate)}
    clone=Clone();raw=json.loads(gzip.decompress((OUT/'branch-breakout-replays.json.gz').read_bytes()))['loss']
    assert raw['match_id']==MATCH
    cfg=raw['configuration'];raw['actions']=raw['actions'][:164];raw['expected']=raw['expected'][:165]
    snaps=history(raw,agents,clone);del raw
    print('prefix verified',cfg['randomSeed'],flush=True)
    progress=a.progress or a.out.with_suffix('.partial.json')
    results=json.loads(progress.read_text()) if progress.exists() else []
    for start,end in START_END:
        for seed in SEEDS:
            for name,m in agents.items():
                if any(r['start']==start and r['rng_seed']==seed and r['arm']==name for r in results):continue
                at=time.monotonic();case=branch(cfg,snaps[start],clone,m,name,start,end,seed);results.append(case)
                tmp=progress.with_suffix('.writing');tmp.write_text(json.dumps(results));tmp.replace(progress)
                print(start,name,seed,case['first'].get('own_seat1'),case['final_cash_margin'],'seconds',round(time.monotonic()-at,1),flush=True)
    calibration=validate_clone(clone,a.cache)
    report=dict(match=MATCH,official_replay_seed=cfg['randomSeed'],official_result_seed=157458776,source_order=['Master7sE','QQ-v6'],frozen_sha256=FROZEN,
                clone_weights={'worker':shot(HERE/'master-worker-model.npz'),'recruit':shot(HERE/'master-model.npz'),'trunk':shot(HERE/'checkpoints/fairlead.py')},
                search='Only own strategies use the frozen 4-wide, 2-deep, one-reply search; as in original turn-190 experiment, at most 3 tail steps; no wall-clock early cut. All other source logic intact. One CPU with low process priority.',
                baseline='Behavior clone, known E worker and recruit heads, freshly conditioned on its own fog and current public positions each step. Original engine coordinated stochastic choice temperature .40. No recorded commands or future observation passed to a branch. Perturbations use common seeds by turn and seat.',
                source_verified_through=164,validation=calibration,cases=results)
    a.out.write_text(json.dumps(report,indent=1)+'\n');print('complete',a.out,flush=True)

if __name__=='__main__':main()
