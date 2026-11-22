"""Exact-referee 18-turn branches for anticipatory grove defense, grouped by layout."""
import copy
import gzip
import hashlib
import json
import os
import sys
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[n]='1'
import numpy as np
from preempt_tree import HERE,MODEL,plan,follow,_v7_pt_profile,_V7_PT_D as D
from branch_replay import step,fields,write
from close_replay import archive_state,restore,referee_events
from benchmark import configure_engine,limit_math_threads,new_game,load_strategy,validate_actions,observe
from arena import engine_fingerprint
ROOT=HERE.parents[1];OUT=ROOT/'analysis/v7';DATA=ROOT.parent/'v7-data'
CP=HERE/'checkpoints/soundings.py';SHA='95d5d869385f83f401aa511b151b8dbdfe692bec02893d1d063465b46dd7247b'
HORIZON=18


def parse_public(r):
    with gzip.open(ROOT/r['source']/'replays'/r['file']) as f:g=json.load(f)
    return g['configuration'],g['steps'],r['layout']


def local_frames(r):
    state,env=new_game(r['seed'],300);cfg=dict(env.configuration);frames=[]
    for t in range(len(r['actions_and_post_cash'])+1):
        acts=r['actions_and_post_cash'][t-1][0] if t else [{},{}]
        if t:
            step(state,env,acts)
            if [p[0] for p in state[0].observation.players]!=r['actions_and_post_cash'][t-1][1]:raise ValueError((r['seed'],t))
        frames.append([dict(observation=dict(v.observation),action=acts[s],reward=v.reward,status=v.status) for s,v in enumerate(state)])
    assert [q.reward for q in state]==r['scores']
    return cfg,frames,r['layout_sha256']


def prospective(frames,sides,coef,key):
    items=[]
    for t in range(35,len(frames)-HORIZON,3):
        current=frames[t][0]['observation'];prior=frames[t-1][0]['observation']['players']
        for s in sides:
            o=dict(current,step=t,player=s)
            # No historical commands, future ownership or outcome enter selection.
            act,intent=plan(o,{'episodeSteps':300},{},prior,coef)
            if intent:
                rank=hashlib.sha256(f'preempt-window/{key}/{t}/{s}'.encode()).hexdigest()
                items.append((rank,t,s,intent))
    return sorted(items)


def diagnostic(frames,side):
    report=[];fs=[f[0]['observation'] for f in frames]
    for t in range(1,len(fs)):
        old,cur=fs[t-1]['players'],fs[t]['players']
        for uid,(p,age) in old[side][3].items():
            if uid not in cur[1-side][3]:continue
            near=min(D[p,q] for q in old[side][1].values())
            rate=fs[t-1]['trees'][uid][1]
            gained=sum(fs[k]['trees'].get(uid,(0,0))[1] for k in range(t,min(len(fs),t+18)) if uid in fs[k]['players'][1-side][3])
            prof=[]
            for k in range(max(1,t-5),t-2):
                o=dict(fs[k],step=k,player=side)
                for row in _v7_pt_profile(o,fs[k-1]['players'][1-side][2],side):
                    if row[0]==uid:prof.append([k,row[6],row[7]])
            report.append(dict(post=t,tree=uid,pos=p,age=age,distance=int(near),pre_income=round(rate,3),
                               subsequent_opponent_gross18=round(gained,2),prior_3_to_5=prof))
    return report


def model(path,name,primary):
    code=path.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)')
    if primary:
        lo=code.index('    if _S_COST[0] > 0.0:',code.index('def _search('));hi=code.index('    stages =',lo)
        code=code[:lo]+'    most,D2,M = 4,min(D2,2),1\n'+code[hi:]
        code=code.replace('for tick in range(1,15):','for tick in range(1,4):').replace('return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT','return False')
    m=load_strategy(name,path,code);m._time=SimpleNamespace(perf_counter=lambda:0.);m._S_MMAX=1;m._torch.set_num_threads(1)
    assert m._SEARCH_OK and m._RX
    m.search_full,m.reply_full=m._search,m._reply_choice
    return m


def budget(m,full):
    m._search=m.search_full if full else lambda o,c,e,a,*rest:a
    m._reply_choice=m.reply_full if full else lambda o,c,e,a,*rest:a


def chooser(m,o,cfg,salt):
    seed=(salt+1000003*(o['step']+1)+47*o['player'])%(2**31)
    m._ST['rng']=np.random.RandomState(seed)
    return m.agent(o,cfg)


def replay(cfg,frames,t,mine,rival,side,salt):
    from zerosum_env.envs.carbon import idgen
    new_game(cfg['randomSeed'],300)
    start=dict(seats=[dict(observation=x['observation'],status=x['status'],reward=x['reward'],action=x['action'] or {}) for x in frames[0]],id_counters=dict(idgen._id_dict))
    state,env=restore(start,cfg)
    public_cfg={k:v for k,v in cfg.items() if k!='randomSeed'}
    mods=[mine,rival] if side==0 else [rival,mine]
    for m in mods:m._ST={};m._FC_HISTORY={};budget(m,False)
    saved=None
    for k,f in enumerate(frames[:t+HORIZON+1]):
        A=[v.observation for v in state];B=[v['observation'] for v in f]
        if fields(A,[v.reward for v in state])!=fields(B,[v['reward'] for v in f]):raise ValueError(('historical mismatch',k))
        if k==t:saved=archive_state(state)
        if k==t+HORIZON:break
        acts=[v['action'] or {} for v in frames[k+1]]
        if k<t:
            ids=dict(idgen._id_dict)
            for s,m in enumerate(mods):
                ob=observe(state,s,k);chooser(m,ob,public_cfg,salt);m._ST['tr'].set_actions(ob['players'],acts[s])
                idgen._id_dict.clear();idgen._id_dict.update(ids)
        step(state,env,acts)
    return saved,mods,copy.deepcopy([(m._ST,m._FC_HISTORY) for m in mods]),public_cfg


def branch(start,mem,mods,config,cfg,t,side,salt,coef,special):
    from zerosum_env.envs.carbon import idgen
    state,env=restore(start,config)
    for m,(st,fc) in zip(mods,copy.deepcopy(mem)):m._ST,m._FC_HISTORY=st,fc
    accounts=[Counter(),Counter()];moves=[];edits=[];marks={};intent=None;initial=None
    target_trees=[];events=[]
    with referee_events(events):
        for delta in range(HORIZON):
            k=t+delta;ids=dict(idgen._id_dict);joint=[]
            for s,m in enumerate(mods):
                ob=observe(state,s,k);prior=m._ST.get('prev');budget(m,delta==0 and s==side)
                a=chooser(m,ob,cfg,salt)
                if s==side:
                    if delta==0:
                        alt,intent=plan(ob,cfg,a,prior,coef);initial=intent
                        if intent:target_trees.append(intent['tree'])
                    elif special and delta<=3 and intent:
                        alt,intent=follow(ob,cfg,a,intent)
                    else:alt=a
                    if special and delta<=3:
                        if alt!=a:edits.append([k,[(u,a.get(u),alt.get(u)) for u in sorted(a.keys()|alt.keys()) if a.get(u)!=alt.get(u)]])
                        a=alt
                validate_actions(a,ob['players'][s]);m._ST['tr'].set_actions(ob['players'],a)
                idgen._id_dict.clear();idgen._id_dict.update(ids);joint.append(a)
            moves.append(joint);events.clear();step(state,env,joint)
            ps=state[0].observation.players;meta=state[0].observation.trees
            for kind,s,am,*rest in events:
                if kind in ('bank','base_collision','plant','seize','tree_yield','hire_planter','hire_collector','tree_collision'):accounts[s][kind]+=am
            for s in (0,1):
                accounts[s]['target_income']+=sum(meta.get(u,(None,0))[1] for u in target_trees if u in ps[s][3])
                accounts[s]['target_owned_turns']+=sum(u in ps[s][3] for u in target_trees)
                if initial and initial['mode']=='exchange':
                    accounts[s]['exchange_site_owned_turns']+=sum(p==initial['goal'] for p,age in ps[s][3].values())
            done=any(v.status!='ACTIVE' for v in state)
            if delta+1 in (5,12,18) or done:
                marks[str(delta+1)]=[dict(cash=p[0],tree_count=len(p[3]),workers=len(p[2]),bank=round(accounts[s]['bank']+accounts[s]['base_collision'],2),
                    tree_income=round(accounts[s]['tree_yield'],2),tree_cost=round(-accounts[s]['plant']-accounts[s]['seize'],2),
                    target_income=round(accounts[s]['target_income'],2),target_owned_turns=accounts[s]['target_owned_turns'],
                    exchange_site_owned_turns=accounts[s]['exchange_site_owned_turns']) for s,p in enumerate(ps)]
            if done:break
    return dict(initial=initial,edits=edits,marks=marks,decisions=moves,turns=len(moves))


def outcomes(cfg,frames,layout,t,s,cohort,coef,mine,rival):
    salt=int(hashlib.sha256(('tree-common/'+layout).encode()).hexdigest()[:8],16)%100000
    snap,mods,mem,vis=replay(cfg,frames,t,mine,rival,s,salt)
    before=branch(snap,mem,mods,cfg,vis,t,s,salt,coef,False)
    if before['initial']:
        after=branch(snap,mem,mods,cfg,vis,t,s,salt,coef,True)
        assert before['initial']==after['initial'] and before['decisions'][0][1-s]==after['decisions'][0][1-s]
    else:after=copy.deepcopy(before)
    delta={}
    for n in ('5','12','18'):
        def mark(r):return r['marks'][str(min(int(n),r['turns']))]
        a,b=mark(after),mark(before)
        delta[n]={k:round((a[s][k]-a[1-s][k])-(b[s][k]-b[1-s][k]),2) for k in a[s]}
    return dict(layout=layout,cohort=cohort,t=t,side=s,initial=before['initial'],edits=after['edits'],
        baseline=before['marks'],candidate=after['marks'],delta=delta,verified_original_frames=t+HORIZON+1,
        initial_referee_id_counts=snap['id_counters'],actions=[before['decisions'],after['decisions']])


def main():
    limit_math_threads();configure_engine();os.environ['CARBON_DEBUG']='1'
    assert hashlib.sha256(CP.read_bytes()).hexdigest()==SHA
    m=model(CP,'preempt_frozen_soundings',True);ref=model(ROOT/'experiments/v6r/checkpoints/estuary.py','preempt_frozen_v6',False)
    reg=json.loads(MODEL.read_text());coef=reg['coef'];cases=[];diagnostics=[];skipped=[]
    files=json.load(open(DATA/'receipts-cache/index.json'))['games'];index={r['layout']:r for r in files}
    def save():
        results={}
        for key in ('local_double_loss','public_heldout'):
            chosen=[c for c in cases if c['cohort']==key];active=[c for c in chosen if c['edits']]
            results[key]=dict(cases=len(chosen),changed=len(active),independent_layouts=len(set(c['layout'] for c in chosen)),
             mean_delta={h:{k:round(float(np.mean([c['delta'][h][k] for c in chosen])),2) for k in ('cash','bank','tree_income','tree_cost','workers','target_owned_turns')} for h in ('5','12','18')} if chosen else {},
             positive_cash18=sum(c['delta']['18']['cash']>0 for c in chosen),negative_cash18=sum(c['delta']['18']['cash']<0 for c in chosen))
        write(OUT/'preempt-tree-branches.json',dict(frozen_sha256=SHA,diagnostics=diagnostics,results=results,skipped=skipped,cases=cases,
            method='Official referee with restored ID counters, legal seat fog replay. First Soundings root uses fixed width 4, depth 2, one response, three-ply tail. Live frozen v6 opponent and both later policies use roots. Maximum four turns pursuing the single assigned worker; then common policies through step 18. Knowledge of subsequent ownership excluded from public window selection.'))
    # Audit each replay's actual transfers, separately from prospective sample selection.
    for name in ('05','06'):
        f=OUT/(name+'-qualifying.jsonl');rows=[json.loads(l) for l in f.open()];by=defaultdict(list)
        for r in rows:by[r['seed']].append(r)
        for seed,rr in by.items():
            def seat(r):return next(i for i,p in enumerate(r['players']) if p['name']!='yan-v6')
            if len(rr)!=2 or any(r['winner']==seat(r) for r in rr):continue
            for r in rr:
                s=seat(r);cfg,frames,k=local_frames(r);events=diagnostic(frames,s)
                opts=prospective(frames,[s],coef,k+'/'+str(s))
                diagnostics.append(dict(candidate=r['players'][s]['name'],seed=seed,seat=s,layout=k,prospective_windows=len(opts),transfers=events))
                if opts:
                    _,t,_,intent=opts[0];cases.append(outcomes(cfg,frames,k,t,s,'local_double_loss',coef,m,ref))
                    print('local',name,seed,s,t,cases[-1]['delta']['18']['cash'],flush=True)
                save()
    used=0
    for k in reg['test_maps']:
        cfg,fs,_=parse_public(index[k]);opts=prospective(fs,[0,1],coef,k)
        if not opts:skipped.append(k);continue
        _,t,s,_=opts[0];cases.append(outcomes(cfg,fs,k,t,s,'public_heldout',coef,m,ref));used+=1
        print('hold',used,t,s,cases[-1]['delta']['18']['cash'],flush=True);save()
        if used>=10:break
    save()


if __name__=='__main__':main()
