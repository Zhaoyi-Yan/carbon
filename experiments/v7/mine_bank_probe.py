"""Test harvest-once-and-bank on independent public maps with a live frozen v6 opponent."""
import os,sys
keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
if __name__=='__main__' and (any(os.environ.get(k)!='1' for k in keys) or os.environ.get('PYTHONHASHSEED')!='0'):
    os.environ.update(dict.fromkeys(keys,'1'),PYTHONHASHSEED='0');os.execv(sys.executable,[sys.executable,*sys.argv])
import copy,gzip,hashlib,json
from collections import Counter
from statistics import mean
from pathlib import Path
from branch_replay import ROOT,HERE,OUT,step,fields,digest
from benchmark import new_game,configure_engine,observe,validate_actions
from close_replay import archive_state,restore,referee_events
from phase_switch import preserved_decision,single_thread,output
from preempt_tree_probe import parse_public,model,budget
from mine_bank import propose,follow

FROZEN={'harbormaster':'d0ba5f631ceeb9ba5670a0013422caca414e407252e6907b90ebe9b33700be87',
        'estuary':'e5c328aa5d18ced597791e7b839088f1a37d18e4e8dc83dd3d54b13d66e877b0'}
FLOWS={'bank','base_collision','tree_collision','plant','seize','tree_yield','rounding','hire_collector','hire_planter'}
MIN_LEVELS=(6.,12.,18.)
hashkey=lambda text:hashlib.sha256(text.encode()).hexdigest()


def cold_window(cfg,fs,k,late):
    offers=[]
    for t in range(90 if late else 12,min(len(fs)-20,235 if late else 90),2):
        for s in (0,1):
            report=fs[t][s]['observation']
            ob=dict(fs[t][0]['observation'],player=s,step=t,carbon=report['carbon'])
            _,i=propose(ob,cfg,fs[t+1][s]['action'] or {},MIN_LEVELS[0])
            if i:offers.append((hashkey(f'mine-bank-window/{k}/{t}/{s}'),t,s))
    return min(offers) if offers else None


def executed(p,o,proposed,actual):
    st=p._ST;st['tr'].set_actions(o['players'],actual)
    if 'liquidity_action' in st:st['liquidity_action']=dict(actual)
    route=st.get('v7_next_depot')
    if route and proposed.get(route[1])!=actual.get(route[1]):st.pop('v7_next_depot')
    if 'corridor_track' in st:
        st['corridor_track']=[e for e in st['corridor_track'] if proposed.get(e[1])==actual.get(e[1])]


def original(cfg,frames,t,side,mods,salt):
    from zerosum_env.envs.carbon import idgen
    new_game(cfg['randomSeed'],300)
    start=dict(seats=[dict(observation=x['observation'],status=x['status'],reward=x['reward'],action=x['action'] or {}) for x in frames[0]],id_counters=dict(idgen._id_dict))
    state,env=restore(start,cfg)
    visible={k:v for k,v in cfg.items() if k!='randomSeed'}
    for p in mods:p._ST={};p._FC_HISTORY={};budget(p,False)
    for i,f in enumerate(frames[:t+21]):
        if fields([v.observation for v in state],[v.reward for v in state])!=fields([v['observation'] for v in f],[v['reward'] for v in f]):
            raise ValueError(('original public replay mismatch',t,side,i))
        if i==t:snap=archive_state(state)
        if i==t+20:break
        actions=[v['action'] or {} for v in frames[i+1]]
        if i<t:
            for s,p in enumerate(mods):
                ob=observe(state,s,i);a=preserved_decision(p,ob,visible,salt);executed(p,ob,a,actions[s])
        step(state,env,actions)
    return snap,copy.deepcopy([(p._ST,p._FC_HISTORY) for p in mods]),visible


def run(cfg,vis,snap,mem,mods,t,side,salt,min_score,treatment):
    state,env=restore(snap,cfg)
    for p,(st,fc) in zip(mods,copy.deepcopy(mem)):p._ST,p._FC_HISTORY=st,fc
    ledger=[Counter(),Counter()];history=[];marks={};changed=[];intent=None;pending=False;deliveries=[]
    start=[v[0] for v in state[0].observation.players]
    for off in range(20):
        current=t+off;actions=[]
        for s,p in enumerate(mods):
            ob=observe(state,s,current);budget(p,off==0)
            normal=preserved_decision(p,ob,vis,salt);out=normal
            if s==side:
                if off==0:
                    alt,intent=propose(ob,vis,normal,min_score)
                    if intent is None:return None
                    if treatment:out=alt;pending=True
                elif treatment and pending:
                    alt,status=follow(ob,vis,normal,intent);pending=status=='return'
                    if pending:out=alt
                    changed.append([current,status,normal.get(intent['uid']),out.get(intent['uid'])])
            validate_actions(out,ob['players'][s]);executed(p,ob,normal,out);actions.append(out)
        history.append(actions);events=[]
        with referee_events(events):step(state,env,actions)
        for kind,s,number,*data in events:
            if kind in FLOWS:ledger[s][kind]+=number
            if kind=='bank' and s==side and data[0]==intent['uid']:deliveries.append([off+1,round(number,3)])
            if kind=='harvest' and s==side and data[0]==intent['uid']:ledger[s]['tracked_mined']+=number
        ps=state[0].observation.players
        if off+1 in (3,5,10,20) or any(s.status!='ACTIVE' for s in state):
            checks=[]
            for i in (0,1):
                assert abs(ps[i][0]-start[i]-sum(v for k,v in ledger[i].items() if k in FLOWS))<.0001
                checks.append(dict(cash=ps[i][0],deposits=round(ledger[i]['bank']+ledger[i]['base_collision'],3),
                    bank_only=round(ledger[i]['bank'],3),tree_income=round(ledger[i]['tree_yield'],3),
                    workers=len(ps[i][2]),tracked_mined=round(ledger[i]['tracked_mined'],3)))
            marks[str(off+1)]=checks
        if any(s.status!='ACTIVE' for s in state):break
    return dict(initial=intent,actions=history,marks=marks,edits=changed,deliveries=deliveries,end=off+1)


def compare(cfg,fs,k,t,s,mods,threshold):
    salt=int(hashkey('mine-bank-common/'+k)[:8],16)%100000;order=mods if not s else list(reversed(mods))
    snap,mem,vis=original(cfg,fs,t,s,order,salt)
    before=run(cfg,vis,snap,mem,order,t,s,salt,threshold,False)
    if before is None:return None
    after=run(cfg,vis,snap,mem,order,t,s,salt,threshold,True)
    assert before['initial']==after['initial'] and before['actions'][0][1-s]==after['actions'][0][1-s]
    assert set(k for k in before['actions'][0][s].keys()|after['actions'][0][s].keys() if before['actions'][0][s].get(k)!=after['actions'][0][s].get(k))=={before['initial']['uid']}
    delta={}
    for h in (3,5,10,20):
        a,b=(r['marks'][str(min(h,r['end']))] for r in (after,before))
        delta[str(h)]={k:round(a[s][k]-a[1-s][k]-(b[s][k]-b[1-s][k]),3) for k in a[s]}
    return dict(layout=k,map_seed=cfg['randomSeed'],start=t,side=s,salt=salt,snapshot_sha256=digest(snap),historical_frames_verified=t+21,
                candidate=after,baseline=before,delta=delta)


def summary(rows):
    if not rows:return {'n':0}
    mean_delta={h:{k:round(mean(r['delta'][h][k] for r in rows),2) for k in ('cash','deposits','bank_only','tree_income','tracked_mined')} for h in ('3','5','10','20')}
    return dict(n=len(rows),mean_delta=mean_delta,banked_by_3=sum(any(t<=3 for t,v in r['candidate']['deliveries']) for r in rows),
         baseline_banked_by_3=sum(any(t<=3 for t,v in r['baseline']['deliveries']) for r in rows),
         positive_cash20=sum(r['delta']['20']['cash']>0 for r in rows),positive_deposits10=sum(r['delta']['10']['deposits']>0 for r in rows))


def main():
    runtime=single_thread();configure_engine();os.environ['CARBON_DEBUG']='1'
    paths=[HERE/'checkpoints/harbormaster.py',ROOT/'experiments/v6r/checkpoints/estuary.py']
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==FROZEN[k] for k,p in zip(FROZEN,paths))
    mods=[model(p,'minebank_'+k,True) for k,p in zip(FROZEN,paths)]
    import torch
    assert torch.get_num_threads()==torch.get_num_interop_threads()==1
    games=json.loads((ROOT.parent/'v7-data/receipts-cache/index.json').read_text())['games']
    excluded={'03100a95e8647c51a4d642fe08d856dd','4063a34afedb6d4767a9118e1dee9e5d','cf276b32e4fd38f43497e60f271a3854'}
    pool=sorted((r for r in games if r['file'].split('-')[0] not in excluded and r['end']>=245),key=lambda r:hashkey('mine-bank-split/'+r['layout']))
    assert len(pool)==len({r['layout'] for r in pool})
    d=dict(frozen=FROZEN,runtime=runtime,policy_sha256=hashlib.sha256((HERE/'mine_bank.py').read_bytes()).hexdigest(),
        rules='Unique public boards, SHA256 split, alternating early/late. At most 28 maps per panel or 12 actual root changes, whichever first. Select one historical candidate across both seats by hash with no future information. At the fork both frozen policies search width 4, depth 2, one counter, three tail steps; then both reactive cheap roots for 20 official transitions. No repeat mission. Development chooses among minimum scores 6,12,18 on paired cash20 and deposits10. Holdout minimum frozen before running.',
        panels={},schedule={s:[r['layout'] for r in pool[offset:offset+28]] for s,offset in [('development',0),('holdout',80)]})
    raw={}
    def save():
        output(OUT/'mine-bank.json',d);output(OUT/'mine-bank-actions.json.gz',raw,True)
    save();chosen=MIN_LEVELS[0]
    for panel,offset in [('development',0),('holdout',80)]:
        results=[];scanned=[];records=d['panels'][panel]=dict(scanned=scanned,results=results,threshold=chosen)
        for i,r in enumerate(pool[offset:offset+28]):
            cfg,fs,k=parse_public(r);win=cold_window(cfg,fs,k,i%2)
            if not win:scanned.append([k,'no_historical_offer']);save();continue
            _,t,s=win
            pair=compare(cfg,fs,k,t,s,mods,chosen)
            if not pair:scanned.append([k,t,s,'no_live_offer']);save();continue
            scanned.append([k,t,s,'active']);raw[k]={a:pair[a].pop('actions') for a in ('baseline','candidate')}
            pair['source_replay']=r['source']+'/replays/'+r['file']
            results.append(pair);records['summary']=summary(results);save()
            print(panel,i+1,len(results),t,s,pair['baseline']['initial']['score'],pair['delta']['10']['deposits'],pair['delta']['20']['cash'],flush=True)
            if len(results)>=12:break
        records['summary']=summary(results)
        if panel=='development':
            cuts={str(level):summary([r for r in results if r['baseline']['initial']['score']>=level]) for level in MIN_LEVELS}
            records['threshold_study']=cuts;poss=[]
            for level in MIN_LEVELS:
                stat=cuts[str(level)]
                if stat['n']>=5 and stat['mean_delta']['10']['deposits']>0 and stat['mean_delta']['20']['cash']>0:
                    quality=stat['n']*(stat['mean_delta']['10']['deposits']+stat['mean_delta']['20']['cash'])/max(1,len(results))
                    poss.append((quality,-level))
            chosen=-max(poss)[1] if poss else MIN_LEVELS[0]
            d['frozen_holdout_score']=chosen;d['development_gate_passed']=bool(poss)
        save()


if __name__=='__main__':main()
