"""Same-midgame, matched-work continuations with two live root replies and an independent v6 seat."""
import os,sys
keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
if __name__=='__main__' and (any(os.environ.get(k)!='1' for k in keys) or os.environ.get('PYTHONHASHSEED')!='0'):
    os.environ.update(dict.fromkeys(keys,'1'),PYTHONHASHSEED='0');os.execv(sys.executable,[sys.executable,*sys.argv])
import copy,gzip,hashlib,json,time
from collections import Counter
from types import SimpleNamespace
from pathlib import Path
from statistics import mean
from branch_replay import ROOT,HERE,OUT,step,board_digest,digest
from benchmark import configure_engine,new_game,load_strategy,observe,validate_actions
from close_replay import archive_state,restore,referee_events
from phase_switch import output,single_thread,preserved_decision
from mine_bank_probe import executed
from preempt_tree_probe import budget
from paired_tail import build,FROZEN

START=185
SOURCES={'development':('v6r-qual-02-canopy.jsonl','canopy'),'holdout':('v6r-qual-03-zenith.jsonl','zenith')}
LIFE=HERE/'checkpoints/lifeline.py';EST=ROOT/'experiments/v6r/checkpoints/estuary.py'
EST_SHA='e5c328aa5d18ced597791e7b839088f1a37d18e4e8dc83dd3d54b13d66e877b0'


def panel():
    out=[]
    for g,(src,cand) in SOURCES.items():
        rows=[json.loads(l) for l in (ROOT/'analysis'/src).open()]
        seeds=sorted({r['seed'] for r in rows},key=lambda n:hashlib.sha256(f'paired-tail-185/{n}'.encode()).digest())
        for i,seed in enumerate(seeds[:4]):
            s=i%2;r,=[r for r in rows if r['seed']==seed and r['players'][s]['name']==cand]
            out.append((g,s,r))
    assert len({r['layout_sha256'] for g,s,r in out})==8
    return out


def load(name,variant=False,production=False,replies=None,extra=None):
    path=EST if name=='v6' else LIFE
    assert hashlib.sha256(path.read_bytes()).hexdigest()==(EST_SHA if name=='v6' else FROZEN)
    src=path.read_text();src=build(src) if variant else src
    edits=[('_torch.set_num_threads(2)','_torch.set_num_threads(1)')]
    if not production:
        lo=src.index('    if _S_COST[0] > 0.0:',src.index('def _search('));hi=src.index('    stages =',lo)
        reply=(1 if name=='v6' else 2) if replies is None else replies;src=src[:lo]+f'    most,D2,M = 4,min(D2,2),{reply}\n'+src[hi:]
        cap=3 if name=='v6' else 8
        edits.extend([('for tick in range(1,15):',f'for tick in range(1,{cap+1}):'),
                      ('return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT','return False')])
        if variant:edits.append(('for _ in range(14//n):','for _ in range(8//n):'))
    for a,b in edits:
        assert src.count(a)==1,a
        src=src.replace(a,b)
    m=load_strategy('paired_tail_'+name,path,src)
    if not production:m._time=SimpleNamespace(perf_counter=lambda:0.);m._S_MMAX=(1 if name=='v6' else 2) if extra is None else extra
    assert m._SEARCH_OK and m._RX;m._torch.set_num_threads(1)
    m.search_full,m.reply_full=m._search,m._reply_choice
    m.audit=Counter()
    m.tail_values=[]
    def tap(fn,label):
        def invoke(*args):
            start=time.process_time();r=fn(*args)
            m.audit[label]+=1;m.audit[label+'_cpu']+=time.process_time()-start;m.audit[label+'_accepted']+=r is not None
            return r
        return invoke
    m._r6_tail=tap(m._r6_tail,'single_tail')
    if variant:m._v7_tail_extend=tap(m._v7_tail_extend,'multi_tail')
    return m


def warm(row,s,mods):
    state,env=new_game(row['seed'],300);cfg=dict(env.configuration);vis={k:v for k,v in cfg.items() if k!='randomSeed'}
    salt=int(hashlib.sha256(('paired-tail-salt/'+row['layout_sha256']).encode()).hexdigest()[:8],16)%100000
    for m in mods:m._ST={};m._FC_HISTORY={};budget(m,False)
    # Frozen and paired variants execute the same root without search during teacher forcing.
    for t,(actions,cash) in enumerate(row['actions_and_post_cash']):
        if t==START:snap=archive_state(state);signature=board_digest(state);mem=copy.deepcopy([(m._ST,m._FC_HISTORY) for m in mods])
        if t<START:
            for p,i in zip(mods,[s,s,1-s]):
                ob=observe(state,i,t);normal=preserved_decision(p,ob,vis,salt);executed(p,ob,normal,actions[i])
        step(state,env,actions)
        assert [p[0] for p in state[0].observation.players]==cash
    assert [p.reward for p in state]==row['scores']
    return cfg,vis,salt,snap,signature,mem


def play(cfg,vis,salt,snap,mem,own,opp,side):
    state,env=restore(snap,cfg);mods=[own,opp] if not side else [opp,own]
    for m,st in zip((own,opp),mem):m._ST,m._FC_HISTORY=copy.deepcopy(st);budget(m,True);m.audit.clear()
    counts=[m._S_STATS['searched'] for m in mods];cpu=[[],[]];wall=[[],[]];bank=[0.,0.];marks={};history=[]
    for t in range(START,299):
        a=[]
        for i,m in enumerate(mods):
            ob=observe(state,i,t);w=time.perf_counter();c=time.process_time()
            act=preserved_decision(m,ob,vis,salt);cpu[i].append(time.process_time()-c);wall[i].append(time.perf_counter()-w)
            validate_actions(act,ob['players'][i]);a.append(act)
        events=[]
        with referee_events(events):step(state,env,a)
        history.append(a)
        for k,i,v,*_ in events:
            if k in ('bank','base_collision'):bank[i]+=v
        if t+1 in (200,225,250,275,299) or any(p.status!='ACTIVE' for p in state):
            marks[str(t+1)]=dict(cash=[p[0] for p in state[0].observation.players],deposits=[round(b,4) for b in bank])
        if any(p.status!='ACTIVE' for p in state):break
    scores=[p.reward for p in state];margin=round(scores[side]-scores[1-side],2)
    assert t+1==299 or all(p.status=='DONE' for p in state)
    return dict(scores=scores,margin=margin,win=int(margin>0)-int(margin<0),end=t+1,marks=marks,hash=board_digest(state),actions_sha256=digest(history),
                searched=[m._S_STATS['searched']-count for m,count in zip(mods,counts)],
                seconds=dict(cpu=[round(sum(v),5) for v in cpu],wall=[round(sum(v),5) for v in wall],peak_cpu=[round(max(v),5) for v in cpu]),
                tail_audit=[dict(m.audit) for m in mods]),history


def main():
    rt=single_thread();configure_engine();os.environ['CARBON_DEBUG']='1'
    rows=panel();models=[load('original'),load('paired',True),load('v6')]
    d=dict(runtime=rt,baseline=FROZEN,opponent=EST_SHA,plugin_sha256=hashlib.sha256((HERE/'paired_tail.py').read_bytes()).hexdigest(),
      schedule=[dict(split=g,side=s,seed=r['seed'],layout=r['layout_sha256']) for g,s,r in rows],start=START,
      budget=dict(focal_width=4,focal_depth=2,focal_root_replies=2,focal_tail_pair_ticks=8,opponent_width=4,opponent_depth=2,opponent_replies=1,opponent_tail_ticks=3,wall_clock=False),
      cases=[])
    actions={}
    def save():output(OUT/'paired-tail.json',d);output(OUT/'paired-tail-actions.json.gz',actions,True)
    save()
    for g,s,r in rows:
        cfg,vis,salt,snap,signature,mem=warm(r,s,models)
        case=dict(split=g,side=s,seed=r['seed'],layout=r['layout_sha256'],initial_hash=signature,initial_id_counters=snap['id_counters'],salt=salt,
            initial_cash=[v[0] for v in snap['seats'][0]['observation']['players']],historical_names=[v['name'] for v in r['players']],arms={})
        d['cases'].append(case);k=f"{r['seed']}/{s}";actions[k]={}
        for j,label in enumerate(('original','paired')):
            v,a=play(cfg,vis,salt,snap,(mem[j],mem[-1]),models[j],models[-1],s)
            case['arms'][label]=v;actions[k][label]=a;save()
            print(g,r['seed'],s,label,v['margin'],v['seconds']['cpu'][s],v['tail_audit'][s],flush=True)


if __name__=='__main__':main()
