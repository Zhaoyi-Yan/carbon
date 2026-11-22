"""Calibrate equal-CPU width/depth alternatives, then verify paired endings on hash-locked historical boards."""
import argparse,copy,gzip,hashlib,json,os,time
from pathlib import Path
import master_e_branch as base
from branch_replay import digest,fields
from benchmark import new_game
from carrier_action_fit import canonical
import numpy as np

MANIFEST=base.ROOT.parent/'v6-data/index.json'
SALT='harbor-horizon-public-v1:'
WIDE=((8,2),(10,2),(12,2))
DEEP=((3,4),(4,4),(5,4))
RIVAL=(4,2)
CAL_STATES=15


def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def key(plan):return f'{plan[0]}x{plan[1]}'
def med(xs):return round(float(np.median(xs)),3)
def label(c):return {k:v for k,v in c.items() if k!='file'}


def panel():
    index=json.loads(MANIFEST.read_text());blocked=set();hist=base.OUT/'carrier-action-factual.json'
    d=json.loads(hist.read_text());blocked.update(d['excluded_canonical'])
    blocked.update(r['canonical'] for k in ('train_maps','test_maps') for r in d[k])
    used=set(blocked);out=[];seats=[0,0]
    for r in sorted(index,key=lambda r:sha(SALT+r['id']+'/'+str(r['game']))):
        if r['steps']!=299 or not r['finished']:continue
        names=[r['strategies'][k]['name'] for k in r['order']]
        if names.count('yan-v5')!=1:continue
        s=names.index('yan-v5');n=seats[s]
        if n>=3:continue
        file=MANIFEST.parent/r['file']
        if not file.is_file():continue
        g=base.read_replay(file);fp=canonical(g['steps'][0][0]['observation'])
        if fp in used:continue
        used.add(fp);seats[s]+=1
        start=((150,150,190),(190,190,150))[s][n]
        out.append(dict(match=r['id'],game=r['game'],seat=s,start=start,original=names,fold='dev' if n==0 else 'holdout',
                        map_seed=g['configuration']['randomSeed'],policy_seed=int(sha('harbor-search-seed-v1:'+r['id'])[:6],16),
                        map_fingerprint=fp,sort_digest=sha(SALT+r['id']+'/'+str(r['game'])),file=str(file)))
    if seats!=[3,3]:raise ValueError(('Not enough maps',seats))
    return sorted(out,key=lambda c:(c['fold']=='holdout',c['sort_digest']))


def load(name,path):
    if base.shot(path)!=base.FROZEN[name]:raise ValueError((name,'hash'))
    src=path.read_text();i=src.index('    if _S_COST[0] > 0.0:',src.index('def _search('));j=src.index('    stages =',i)
    src=src[:i]+'    most,D2,M=_BUDGET_PLAN[0],min(D2,_BUDGET_PLAN[1]),1\n'+src[j:]
    for old,new in [
        ('+ 17 * me) % (2 ** 31)','+ 17 * me + _BRANCH_SEED) % (2 ** 31)'),
        ('for tick in range(1,15):','for tick in range(1,4):'),('_torch.set_num_threads(2)','_torch.set_num_threads(1)'),
        ('return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT','return False'),
        ('import time as _time','from types import SimpleNamespace as _ClockNS\n_time=_ClockNS(perf_counter=lambda:0.)')]:
        if src.count(old)!=1:raise ValueError((name,old))
        src=src.replace(old,new)
    m=base.load_strategy('width_depth_'+name,path,src)
    m._S_MMAX=1;m._BRANCH_SEED=0;m._BUDGET_PLAN=RIVAL
    assert m._SEARCH_OK and m._torch.get_num_threads()==1
    step=m._turn;m._measured_steps=0
    def measured(*args,**kwargs):
        m._measured_steps+=1
        return step(*args,**kwargs)
    m._turn=measured
    return m


def memory(m):return copy.deepcopy((m._ST,m._FC_HISTORY))
def reset(m,mem):m._ST,m._FC_HISTORY=copy.deepcopy(mem)
def state_info(state):
    return [base.readings(state[0].observation.players,s) for s in (0,1)]


def prepare(c,ours,opp,calibration):
    from zerosum_env.envs.carbon import idgen
    g=base.read_replay(Path(c['file']));cfg=g['configuration'];rules={k:v for k,v in cfg.items() if k!='randomSeed'}
    s=c['seat'];start=c['start']
    new_game(cfg['randomSeed'],cfg['episodeSteps'])
    first=dict(seats=[{k:copy.deepcopy(a[k]) for k in ('observation','reward','status','action')} for a in g['steps'][0]],
               id_counters=dict(idgen._id_dict))
    state,env=base.restore(first,cfg)
    samples=[];times=set(start-42+3*i for i in range(CAL_STATES)) if calibration else set()
    agents={s:ours,1-s:opp};save={i:(m._search,m._reply_choice) for i,m in agents.items()}
    for m in agents.values():
        m._ST={};m._FC_HISTORY={}
        m._search=m._reply_choice=lambda obs,cfg,est,act,*rest:act
    for t in range(start+1):
        real=g['steps'][t]
        assert base.board_digest(state)==digest(fields([x['observation'] for x in real],[x['reward'] for x in real]))
        if t in times:samples.append((t,copy.deepcopy(base.observe(state,s,t)),memory(ours)))
        if t==start:break
        commands=[v['action'] or {} for v in g['steps'][t+1]];ids=dict(idgen._id_dict)
        for i,m in agents.items():
            view=base.observe(state,i,t);sample=m.agent(view,rules);cmd=commands[i]
            m._ST['tr'].set_actions(view['players'],cmd)
            if 'liquidity_action' in m._ST:m._ST['liquidity_action']=dict(cmd)
            if 'corridor_track' in m._ST:
                m._ST['corridor_track']=[p for p in m._ST['corridor_track'] if sample.get(p[1])==cmd.get(p[1])]
        idgen._id_dict.clear();idgen._id_dict.update(ids);base.step(state,env,commands)
    for i,m in agents.items():m._search,m._reply_choice=save[i]
    return dict(snapshot=base.archive_state(state),cfg=cfg,source_holdings=state_info(state),mem=[memory(ours),memory(opp)],samples=samples)


def act(m,obs,cfg,plan,seed):
    t=obs['step'];s=obs['player'];m._BUDGET_PLAN=plan;m._BRANCH_SEED=seed
    m._ST['rng']=np.random.RandomState((seed+1000003*(t+1)+s)%(2**32))
    before=m._S_STATS['searched'];m._measured_steps=0
    at=time.process_time_ns();cmd=m.agent(obs,cfg);cpu=(time.process_time_ns()-at)/1.e6
    last=m._S_STATS.get('last') if m._S_STATS['searched']>before else None
    return cmd,[round(cpu,3),m._measured_steps,int(last[0]) if last else 0,int(last[1]) if last else 0]


def calibrate(dev,m):
    from zerosum_env.envs.carbon import idgen
    plans=WIDE+DEEP;raw={key(p):[] for p in plans};pairs=[]
    for c,pack in dev:
        rules={k:v for k,v in pack['cfg'].items() if k!='randomSeed'}
        for t,obs,mem in pack['samples']:
            order=sorted(plans,key=lambda p:sha(c['match']+str(t)+key(p)));ids=dict(idgen._id_dict)
            for p in order:
                reset(m,mem);cmd,cost=act(m,obs,rules,p,c['policy_seed']);base.validate_actions(cmd,obs['players'][c['seat']])
                idgen._id_dict.clear();idgen._id_dict.update(ids)
                raw[key(p)].append([c['match'],t,*cost])
    stats={k:{'n':len(v),'cpu_ms':med([r[2] for r in v]),'simulated_turns':med([r[3] for r in v]),
              'root_actions':med([r[4] for r in v]),'achieved_depth':med([r[5] for r in v])} for k,v in raw.items()}
    for a in WIDE:
        for b in DEEP:
            wide,deep=stats[key(a)],stats[key(b)]
            cpu=wide['cpu_ms']/deep['cpu_ms'];sim=wide['simulated_turns']/deep['simulated_turns']
            pairs.append((abs(np.log(cpu)),abs(np.log(sim)),a,b,cpu,sim))
    eligible=[v for v in pairs if max(v[4],1/v[4])<=1.15]
    if not eligible:raise ValueError(('No CPU-matched grid',stats))
    # Use CPU (primary) and physics updates (cross-check), without reading any game continuation.
    x=min(eligible,key=lambda p:p[0]+.3*p[1])
    return x[2],x[3],dict(states=2*CAL_STATES,rows=raw,columns=['match','turn','process_cpu_ms','simulated_board_transitions','root_actions','reached_depth'],
                median=stats,rule='Among pairs within 15% median process CPU, minimize |log(cpu ratio)| + 0.3 |log(simulated-turn ratio)|. No continuation outcomes available.',
                choices=[key(x[2]),key(x[3])],cpu_ratio=x[4],sim_ratio=x[5])


def play(c,pack,ours,opp,plan):
    from zerosum_env.envs.carbon import idgen
    cfg=pack['cfg'];rules={k:v for k,v in cfg.items() if k!='randomSeed'}
    state,env=base.restore(pack['snapshot'],cfg);s=c['seat'];t0=c['start'];seed=c['policy_seed']
    reset(ours,pack['mem'][0]);reset(opp,pack['mem'][1]);costs=[];enemy_cost=[];events=[];marks={};h=hashlib.sha256()
    for t in range(t0,299):
        joint=[None,None];ids=dict(idgen._id_dict)
        for side,agent,budget,meter in ((s,ours,plan,costs),(1-s,opp,RIVAL,enemy_cost)):
            view=base.observe(state,side,t);assert 'full_carbon' not in view
            cmd,d=act(agent,view,rules,budget,seed);base.validate_actions(cmd,view['players'][side])
            idgen._id_dict.clear();idgen._id_dict.update(ids);joint[side]=cmd;meter.append(d)
        h.update(json.dumps(joint,sort_keys=True,separators=(',',':')).encode()+b'\n')
        for recipient,owner,uid,amount in base.deposits(state[0].observation.players,joint):
            events.append([t+1,recipient,owner,uid,round(amount,4)])
        base.step(state,env,joint)
        if t+1 in (t0+10,t0+20,299) or any(a.status=='DONE' for a in state):
            holdings=state_info(state)
            own=sum(v for _,r,o,_,v in events if r==o==s);foe=sum(v for _,r,o,_,v in events if r==o==1-s)
            marks[t+1]=dict(seats=holdings,own_gross=round(own,3),opp_gross=round(foe,3),
                            net_gross=round(own-foe,3),cash_margin=round(holdings[s]['cash']-holdings[1-s]['cash'],2))
        if any(a.status=='DONE' for a in state):break
    final=t+1;reward=[a.reward for a in state]
    return dict(plan=key(plan),from_turn=t0,through=final,seat=s,source=pack['source_holdings'],map_seed=c['map_seed'],policy_seed=seed,
                milestones=marks,reward=reward,result=int(np.sign(reward[s]-reward[1-s])),margin=marks[final]['cash_margin'],
                costs=costs,opponent_costs=enemy_cost,events=events,actions_sha256=h.hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',required=True,type=Path);p.add_argument('--locked-result',type=Path,help='Reuse a published panel and its calibrated schedules when reproducing trajectories');p.add_argument('--out',default=base.OUT/'search-budget.json',type=Path);a=p.parse_args();a.work.mkdir(exist_ok=True,parents=True)
    base.limit_math_threads();base.configure_engine();os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    panel_rows=panel();public=[label(c) for c in panel_rows]
    pp=a.work/'fixed-panel.json'
    if pp.exists():assert json.loads(pp.read_text())==public
    else:pp.write_text(json.dumps(public,indent=1)+'\n')
    print('locked',[(c['fold'],c['match'][:8],c['seat'],c['start']) for c in panel_rows],flush=True)
    ours=load('harbormaster',base.HERE/'checkpoints/harbormaster.py');rival=load('v6',base.ROOT/'yan-v6.py')
    dev=[(c,prepare(c,ours,rival,True)) for c in panel_rows if c['fold']=='dev']
    cp=a.work/'calibration.json'
    if a.locked_result:
        locked=json.loads(a.locked_result.read_text());assert locked['panel']==public and locked['sha256']==base.FROZEN
        if cp.exists():assert json.loads(cp.read_text())==locked['calibration']
        else:cp.write_text(json.dumps(locked['calibration'],indent=1)+'\n')
    if cp.exists():
        calibration=json.loads(cp.read_text());wide,deep=[tuple(map(int,z.split('x'))) for z in calibration['choices']]
    else:
        wide,deep,calibration=calibrate(dev,ours);cp.write_text(json.dumps(calibration,indent=1)+'\n')
    print('calibrated',calibration['choices'],calibration['cpu_ratio'],calibration['sim_ratio'],calibration['median'],flush=True)
    devpacks={c['match']:p for c,p in dev}
    for item in devpacks.values():item.pop('samples')
    rows=[]
    for c in panel_rows:
        continuation=a.work/(c['match']+'.json')
        if continuation.exists():
            row=json.loads(continuation.read_text());assert set(row['arms'])=={key(wide),key(deep)};rows.append(row);continue
        pack=devpacks.pop(c['match'],None) or prepare(c,ours,rival,False)
        runs={};order=(wide,deep) if int(c['sort_digest'][-2:],16)%2==0 else (deep,wide)
        for p_ in order:
            run=play(c,pack,ours,rival,p_);runs[key(p_)]=run
            print('game',c['fold'],c['match'][:8],run['plan'],run['margin'],run['result'],med([v[0] for v in run['costs']]),flush=True)
        row={'map':label(c),'arms':runs};continuation.write_text(json.dumps(row)+'\n');rows.append(row)
    out=dict(panel=public,panel_rule='Completed cached yan-v5 historical matches, 299 transitions; exclude prior carrier-action groupings and both known fork boards; SHA256 '+SALT+'match/game; take first three disjoint canonical layouts per QQ seat, first develop, other two held out. Four holdouts balanced 2/2 by start and seat.',
             sha256=base.FROZEN,opponent='frozen v6; fixed 4x2/one modeled reply/3 tail transitions in both arms; live independent response to changed boards',
             experiment='Frozen Harbormaster; only root width and depth change, both use one sampled enemy response, up to 3 fixed tail steps on finalists and one rollout. Strategy timers fixed independently of process CPU measurements.',
             jitter='Shared per-turn, per-seat random numbers in both continuations. Costs are single thread process CPU milliseconds and explicitly counted model transitions.',
             calibration=calibration,results=rows)
    a.out.write_text(json.dumps(out,separators=(',',':'))+'\n');print('done',a.out,flush=True)

if __name__=='__main__':main()
