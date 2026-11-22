"""Check early terminals, duplicate and partial replies, then inspect live production search tails."""
import os,sys
keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
if __name__=='__main__' and (any(os.environ.get(k)!='1' for k in keys) or os.environ.get('PYTHONHASHSEED')!='0'):
    os.environ.update(dict.fromkeys(keys,'1'),PYTHONHASHSEED='0');os.execv(sys.executable,[sys.executable,*sys.argv])
import argparse,copy,gzip,hashlib,inspect,json,time
from collections import Counter
from types import SimpleNamespace
from statistics import mean,median
import numpy as np
from branch_replay import ROOT,HERE,OUT,step,digest,board_digest
from benchmark import configure_engine,new_game,observe,validate_actions
from close_replay import restore,archive_state,referee_events
from phase_switch import output,single_thread,preserved_decision
from preempt_tree_probe import budget
from paired_tail_probe import START,SOURCES,panel,load,warm
import paired_tail


def fixtures():
    timer=SimpleNamespace(now=0.);timer.perf_counter=lambda:timer.now
    draws=[];terminal_hits=[]
    def sample(pl,s,w,b,rng,t):
        draws.append((s,t,int(rng.randint(1<<24))));return {}
    def tiny(v,ids):
        assert len(ids)==len(v);a=np.zeros((len(v),1));return a,a,None
    def turn(b,c,a):
        assert b.get('_terminal_sign') is None
        result=dict(b,step=b['step']+1,v=b['v']+1.)
        if b.get('stop')==result['step']:result['_terminal_sign']=-1;terminal_hits.append(result['step'])
        return result
    def legacy(*args):return ('legacy',args[0])
    env=dict(np=np,LAST_TURN=299,_time=timer,_v5_pair=lambda *_:(0.,1.),_sample=sample,
             _V6R_TINY=tiny,_v7_rollout_ids=lambda x:x,_turn=turn,_terminal=float,
             _r6_asset_score=lambda b,s:b['v'],_r6_tail=legacy)
    for fn in (paired_tail._v7_tail_over,paired_tail._v7_tail_pairs,paired_tail._v7_tail_extend,paired_tail._v7_paired_tail):
        exec(inspect.getsource(fn),env)
    def row(c,m,s=22,v=0.,**kw):return [c,m,dict(step=s,v=v,carbon=[0.],players=[[1],[1]],**kw),None]
    pair=env['_v7_tail_pairs'];call=env['_v7_paired_tail'];A={0:{'u':'UP'},1:{'u':'DOWN'}}
    rows=[row(0,0,v=4),row(1,0,v=-2),row(0,1,v=20),row(1,1,v=30)]
    tags=[0,1];args=(tags,0,{},0.,41,.83)
    assert call(rows,[],*args,{0:{}},22)==('legacy',rows)
    assert call(rows,rows,*args,{0:{'u':'UP'},1:{'u':'UP','unused':None}},22)==('legacy',rows)
    assert len(pair(rows[:-1],tags,A,22))==1
    assert len(pair(rows[:2]+[row(0,1,s=21),row(1,1,s=23)],tags,A,22))==1
    assert len(pair(rows[:2]+[row(0,1,s=20,_terminal_sign=1),row(1,1,s=22)],tags,A,22))==2
    assert call(rows,rows,*args,A,22)==[19.,21.]  # Seven ticks per group; average both replies.
    assert all(len({d[2] for d in draws if d[:2]==(s,t)})==1 for t in range(22,29) for s in (0,1))
    rows=[row(0,0,stop=24),row(1,0),row(0,1,s=20,_terminal_sign=1),row(1,1)]
    assert call(rows,rows,*args,A,22)==[0.,7.] and terminal_hits==[24]
    groups=pair([row(0,0),row(1,0),row(0,1),row(1,1)],tags,A,22)
    def fail(*a):raise ValueError('injected')
    env['_turn']=fail
    assert env['_v7_tail_extend'](groups,0,{},0,4,.83) is None
    counter=[]
    def interrupt(*a):
        counter.append(1)
        if len(counter)==2:timer.now=.90
        return turn(*a)
    env['_turn']=interrupt
    assert env['_v7_tail_extend'](groups,0,{},0,4,.83) is None
    assert all(st[2]['step']==22 for group in groups for st in group)
    return dict(single_legacy=True,duplicate_reply_discarded=True,incomplete_reply_discarded=True,scored_depth_checked=True,
        partial_tick_atomic=True,exception_falls_back=True,terminal_is_absorbing=True,common_draws=True)


def record_module(m,variant):
    m.records=[]
    if variant:
        orig=m._v7_paired_tail
        def paired(legacy,scored,candidates,side,cfg,t0,seed,allow,opp,depth):
            gs=m._v7_tail_pairs(scored,candidates,opp,depth);before=set(r[1] for r in legacy)
            rec=dict(replies=len(opp),unique_pairs=len(gs),scored_ids=sorted({r[1] for r in scored}),legacy_ids=sorted(before),
                appended=[p[0][1] for p in gs if p[0][1] not in before],depth_step=depth)
            v=orig(legacy,scored,candidates,side,cfg,t0,seed,allow,opp,depth);rec['tail_values']=v;m.records.append(rec);return v
        m._v7_paired_tail=paired
    old=m._shadow_reply;m.shadow=0
    def shadow(*args):m.shadow+=1;return old(*args)
    m._shadow_reply=shadow


def command(m,mem,snap,cfg,visible,side,salt):
    state,_=restore(snap,cfg);m._ST,m._FC_HISTORY=copy.deepcopy(mem);budget(m,True)
    o=observe(state,side,START);m.records.clear();m.audit.clear();m.shadow=0
    w=time.perf_counter();t=time.process_time();a=preserved_decision(m,o,visible,salt)
    return a,dict(cpu=round(time.process_time()-t,5),wall=round(time.perf_counter()-w,5),
                  searched=m._S_STATS.get('last',())[:3],root_shadow_calls=m.shadow,tail=dict(m.audit),records=copy.deepcopy(m.records))


def production():
    models=[load('prod_base',production=True),load('prod_pair',True,production=True),load('v6')]
    singles=[load('one',replies=1,extra=1),load('one_paired',True,replies=1,extra=1)]
    extra=load('extra',True,replies=1,extra=2)
    for m,v in [(models[0],False),(models[1],True),(singles[0],False),(singles[1],True),(extra,True)]:record_module(m,v)
    samples=[]
    for ii,(split,s,r) in enumerate(panel()):
        if ii not in (0,1,4,5):continue
        cfg,vis,salt,snap,_,mem=warm(r,s,models)
        item=dict(split=split,side=s,seed=r['seed'],turn=START)
        old,old_record=command(models[0],mem[0],snap,cfg,vis,s,salt)
        new,new_record=command(models[1],mem[1],snap,cfg,vis,s,salt)
        assert old_record['root_shadow_calls']==new_record['root_shadow_calls']==1
        item.update(original=old_record,paired=new_record,same_action=old==new)
        if ii in (0,4):
            a,ac=command(singles[0],mem[0],snap,cfg,vis,s,salt);b,bc=command(singles[1],mem[0],snap,cfg,vis,s,salt)
            assert a==b and ac['root_shadow_calls']==bc['root_shadow_calls']==1
            assert bc['records'] and bc['records'][0]['unique_pairs']==1
            item.update(single_original=ac,single_paired=bc,single_equal=True)
        _,ec=command(extra,mem[0],snap,cfg,vis,s,salt)
        assert ec['records'] and ec['records'][0]['replies']==2
        assert 1 in ec['records'][0]['scored_ids'] and 1 not in ec['records'][0]['legacy_ids']
        item['forced_extra_reply']=ec
        samples.append(item)
    assert any(q['forced_extra_reply']['records'][0]['appended']==[1] for q in samples)
    return samples


def replay():
    d=json.load(open(OUT/'paired-tail.json'));raw=json.loads(gzip.decompress((OUT/'paired-tail-actions.json.gz').read_bytes()))
    assert len(d['cases'])==8 and d['plugin_sha256']==hashlib.sha256((HERE/'paired_tail.py').read_bytes()).hexdigest()
    groups={};divergence={}
    for c,(g,s,r) in zip(d['cases'],panel()):
        assert (c['split'],c['side'],c['seed'])==(g,s,r['seed'])
        st,env=new_game(c['seed'],300);cfg=dict(env.configuration)
        for a,check in r['actions_and_post_cash'][:START]:
            step(st,env,a);assert [p[0] for p in st[0].observation.players]==check
        assert board_digest(st)==c['initial_hash'];snap=archive_state(st);assert snap['id_counters']==c['initial_id_counters']
        traces=raw[f"{c['seed']}/{s}"]
        change=next((t for t,(a,b) in enumerate(zip(traces['original'],traces['paired'])) if a!=b),None)
        if change is not None:
            aa,bb=(traces[key][change] for key in ('original','paired'))
            assert aa[s]!=bb[s] and aa[1-s]==bb[1-s]
        divergence[c['seed']]=None if change is None else START+change
        for name,v in c['arms'].items():
            st,env=restore(snap,cfg);a=raw[f"{c['seed']}/{s}"][name];assert digest(a)==v['actions_sha256']
            bank=[0.,0.]
            for j,acts in enumerate(a,START+1):
                for i,ac in enumerate(acts):validate_actions(ac,st[0].observation.players[i])
                ev=[]
                with referee_events(ev):step(st,env,acts)
                for label,i,value,*_ in ev:
                    if label in ('bank','base_collision'):bank[i]+=value
                if str(j) in v['marks']:
                    x=v['marks'][str(j)]
                    assert x['cash']==[p[0] for p in st[0].observation.players] and x['deposits']==[round(z,4) for z in bank]
            assert [p.reward for p in st]==v['scores'] and v['hash']==board_digest(st)
    for g in ('development','holdout','all'):
        cs=[c for c in d['cases'] if c['split']==g or g=='all'];arm={}
        for name in ('original','paired'):
            z=[c['arms'][name] for c in cs];dur=sum(r['end']-START for r in z)
            cpu=sum(r['seconds']['cpu'][c['side']] for c,r in zip(cs,z));wall=sum(r['seconds']['wall'][c['side']] for c,r in zip(cs,z));bank=[];for_bank=[];against_bank=[]
            audit=Counter()
            for c,r in zip(cs,z):
                s=c['side'];b=r['marks'][str(r['end'])]['deposits'];bank.append(b[s]-b[1-s]);for_bank.append(b[s]);against_bank.append(b[1-s]);audit.update(r['tail_audit'][s])
            arm[name]=dict(wins=sum(r['win']==1 for r in z),losses=sum(r['win']==-1 for r in z),ties=sum(r['win']==0 for r in z),
                mean_margin=round(mean(r['margin'] for r in z),2),mean_deposit_margin=round(mean(bank),2),mean_own_deposits=round(mean(for_bank),2),mean_enemy_deposits=round(mean(against_bank),2),
                cpu_seconds=round(cpu,3),cpu_ms_per_turn=round(1000*cpu/dur,2),wall_ms_per_turn=round(1000*wall/dur,2),tail=audit)
        deltas=[round(c['arms']['paired']['margin']-c['arms']['original']['margin'],2) for c in cs]
        gains=sum(c['arms']['paired']['win']>c['arms']['original']['win'] for c in cs);losses=sum(c['arms']['paired']['win']<c['arms']['original']['win'] for c in cs)
        groups[g]=dict(arms=arm,delta=deltas,mean=round(mean(deltas),2),median=round(median(deltas),2),favorable_results=gains,unfavorable_results=losses,first_focal_divergence=[divergence[c['seed']] for c in cs])
    return groups


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--production',action='store_true');ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    runtime=single_thread();configure_engine();os.environ['CARBON_DEBUG']='1';checks=fixtures()
    if a.production:output(OUT/'paired-tail-production.json',dict(fixtures=checks,samples=production(),runtime=runtime))
    if a.replay:output(OUT/'paired-tail-summary.json',dict(fixtures=checks,verified_suffixes=16,groups=replay()))
    print(checks)
