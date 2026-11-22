"""Compare the one-miner rescue with the frozen baseline on paired development layouts."""
import json
import os
from collections import Counter
from statistics import mean
import income_loop as policy
from opening_probe import HERE,OUT,ROOT,map_id,load_policy,sha
from branch_replay import step,write
from benchmark import configure_engine,limit_math_threads,new_game,observe,validate_actions
from close_replay import referee_events

MARKS=(20,60,100,155,200,299)
OUTPUT=OUT/'income-loop-probe.json'


def gate_counts(obs,cfg,mem,c):
    t=obs['step'];s=obs['player'];p,opp=obs['players'][s],obs['players'][1-s]
    if not 5<=t<=155:return
    c['window']+=1
    price=cfg['recCollectorCost']
    if p[0]>=price:return
    c['cash_below_one_hire']+=1
    if len(p[2])>=8:return
    c['also_fewer_than_eight_workers']+=1
    reserve=p[0]+4*sum(obs.get('trees',{}).get(k,(None,0))[1] for k in p[3])
    if reserve>=price:return
    c['also_short_after_tree_income']+=1
    homes=policy._V7_IL_D[list(p[1].values())].min(0)
    if any(k=='COLLECTOR' and v>=price-p[0] and v>=22 and homes[q]<=4 for q,v,k in p[2].values()):return
    c['also_no_nearby_fundable_load']+=1
    fl=policy._v7_receipts(obs['players'],t,cfg['episodeSteps']-2)
    if reserve+fl[s,0]>=price:return
    c['also_short_after_expected_five_step_bank']+=1
    recent=policy.np.sum(mem['flows'],axis=0)
    past=recent[:,0]+.6*recent[:,1]
    if past[1-s]-past[s]<10 and fl[1-s,0]-fl[s,0]<12:return
    c['also_behind_receipts_or_harvest']+=1
    if not any(k=='COLLECTOR' and load<=14 and homes[q]<=6 for q,load,k in p[2].values()):return
    c['also_eligible_light_collector']+=1
    if mem.get('mission'):c['also_assigned_safe_site']+=1


def run(model,seed,arm,end_at=299,choose=None,inspect_gate=gate_counts,marks_at=MARKS):
    import numpy as np
    from zerosum_env.envs.carbon import idgen
    choose=choose or policy._v7_income_loop
    state,env=new_game(seed,300);cfg=dict(env.configuration);cfg.pop('randomSeed')
    layout=map_id(state)
    memory=[({},{}),({},{})];mon=[{},{}];flow=[Counter(),Counter()]
    gates=[Counter(),Counter()];edits=[[],[]];events=[[],[]];deposits=[[],[]];hires=[[],[]];marks={};ledger=[]
    with referee_events(ledger):
        for t in range(end_at):
            ids=dict(idgen._id_dict);joint=[]
            for s in (0,1):
                ob=observe(state,s,t)
                model._ST,model._FC_HISTORY=memory[s]
                model._ST['rng']=np.random.RandomState((seed+1000003*(t+1)+31*s)%(2**31))
                act=model.agent(ob,cfg)
                if s==arm or (arm is None and not edits[s]):
                    alt=choose(ob,cfg,model.mine_est,act,mon[s])
                    if arm is None and inspect_gate:inspect_gate(ob,cfg,mon[s],gates[s])
                    event=mon[s].get('event')
                    if event:events[s].append([t,*event])
                    if alt!=act:
                        changed=[k for k in sorted(act.keys()|alt.keys()) if act.get(k)!=alt.get(k)]
                        assert len(changed)==1 and changed[0] in ob['players'][s][2]
                        u=changed[0];edits[s].append([t,u,act.get(u),alt.get(u)])
                    if s==arm:act=alt
                validate_actions(act,ob['players'][s]);joint.append(act)
                model._ST['tr'].set_actions(ob['players'],act)
                idgen._id_dict.clear();idgen._id_dict.update(ids)
            ledger.clear();step(state,env,joint)
            for k,s,amount,*rest in ledger:
                if k in ('bank','base_collision','tree_yield','harvest','hire_collector','hire_planter'):
                    flow[s][k]+=amount
                if k=='bank' and amount>0:deposits[s].append([t+1,rest[0],rest[1],round(amount,2)])
                if k.startswith('hire_') and amount<0:hires[s].append([t+1,k])
            end=any(p.status!='ACTIVE' for p in state) or t+1==end_at
            if t+1 in marks_at or end:
                marks[str(t+1)]=[dict(cash=p[0],workers=len(p[2]),trees=len(p[3]),cargo=round(sum(q[1] for q in p[2].values()),2),
                                      inbank=round(flow[s]['bank']+flow[s]['base_collision'],2),harvest=round(flow[s]['harvest'],2),
                                      recruit_count=len(hires[s]),new_collectors=sum(k=='hire_collector' for _,k in hires[s]),
                                      tree_income=round(flow[s]['tree_yield'],2),reward=state[s].reward) for s,p in enumerate(state[0].observation.players)]
            if end:break
    return dict(seed=seed,seat=arm,layout=layout,turns=t+1,marks=marks,gates=gates,events=events,edits=edits,receipts=deposits,hires=hires)


def summary(controls,candidates):
    rows=[]
    for c in candidates:
        s=c['seat'];control=controls[c['seed']]
        assignment=[(t,uid,q) for t,what,uid,q in c['events'][s] if what=='assign']
        contracts=[]
        for t,uid,pos in assignment:
            stop=next((u for u,k,w,v in c['events'][s] if u>t and k=='stop' and uid==w),t+13)
            delivered=next((d for d in c['receipts'][s] if t<d[0]<=min(t+13,stop+1) and d[1]==uid and d[2]==s),None)
            hire=next((v for v in c['hires'][s] if delivered and delivered[0]<v[0]<=delivered[0]+4),None)
            contracts.append(dict(turn=t,uid=uid,tile=pos,delivered=delivered,next_hire_within_four_steps=hire))
        marks={}
        for t in MARKS:
            now=c['marks'][str(min(t,c['turns']))];old=control['marks'][str(min(t,control['turns']))]
            marks[str(t)]={k:round(now[s][k]-old[s][k],2) for k in ('cash','workers','trees','inbank','harvest','recruit_count')}
            marks[str(t)]['margin_delta']=round(now[s]['cash']-now[1-s]['cash']-old[s]['cash']+old[1-s]['cash'],2)
        score=c['marks'][str(c['turns'])];margin=round(score[s]['reward']-score[1-s]['reward'],2)
        rows.append(dict(seed=c['seed'],seat=s,changed=len(c['edits'][s]),contracts=contracts,final_margin=margin,delta=marks))
    return dict(rows=rows,wins=sum(x['final_margin']>0 for x in rows),wins_by_seat=[sum(x['final_margin']>0 for x in rows if x['seat']==i) for i in (0,1)],
                improved=sum(x['delta']['299']['margin_delta']>0 for x in rows),change_cases=sum(x['changed']>0 for x in rows),
                average_deltas={str(t):{k:round(mean(r['delta'][str(t)][k] for r in rows),2) for k in rows[0]['delta'][str(t)]} for t in MARKS})


def main():
    import numpy as np
    import receipts
    os.environ['CARBON_DEBUG']='1';limit_math_threads();configure_engine()
    model=load_policy()
    original=model._search
    def capture(obs,cfg,est,*a):
        model.mine_est=est
        return original(obs,cfg,est,*a)
    model._search=capture
    receipts._V7_RECEIPT_DIST=tuple(tuple(v for v in row) for row in model._V5_DIST)
    receipts._V7_RECEIPT_STEPS=np.arange(1,11)
    with np.load(receipts.MODEL,allow_pickle=False) as f:receipts._V7_RECEIPT_MODEL=dict(f)
    policy._V7_IL_D=np.asarray(receipts._V7_RECEIPT_DIST);policy._V7_IL_L=np.asarray([[model.move(p,a) for a in range(5)] for p in range(225)])
    policy.WIDX,policy.WACT,policy._v7_receipts=model.WIDX,model.WACT,receipts._v7_receipts
    import sys
    if '--smoke' in sys.argv:
        # A fixed, hand-constructed legal observation sequence tests the state machine only.
        trace=[];mem={};cfg={'episodeSteps':300,'recCollectorCost':30}
        for t,pos,load,stock,cash in [(14,16,0,100,5),(15,16,30,70,5),(16,16,51,49,5),(17,1,51,0,5),(18,0,0,0,56)]:
            pl=[[cash,{'a':0,'b':30},{'miner':[pos,load,'COLLECTOR']},{}],[100,{'c':195,'d':202},{'truck':[180,140,'COLLECTOR']},{}]]
            o=dict(step=t,player=0,players=pl,trees={},carbon=np.zeros(225));o['carbon'][pos]=stock
            act=policy._v7_income_loop(o,cfg,o['carbon'],{'miner':'RIGHT'},mem)
            trace.append([t,act.get('miner','STAY')])
        assert [r[1] for r in trace]==['STAY','STAY','UP','LEFT','RIGHT'] and 'mission' not in mem
        report=json.loads(OUTPUT.read_text());report['synthetic_finite_state_check']=trace;write(OUTPUT,report);print(trace)
        return
    if '--gates' in sys.argv:
        r=json.loads(OUTPUT.read_text())
        for seed,v in r['controls'].items():
            check=run(model,int(seed),None,156)
            for t in ('20','60','100','155'):
                assert v['marks'][t]==check['marks'][t]
            v['gates']=check['gates']
        for candidate in r['candidates']:
            if candidate['identical_without_edit']:candidate['gates']=r['controls'][str(candidate['seed'])]['gates']
        write(OUTPUT,r)
        print(sum((Counter(g) for v in r['controls'].values() for g in v['gates']),Counter()),flush=True)
        return
    reference=json.loads((OUT/'opening-probe.json').read_text())
    previous={r['seed']:r for r in reference['records'] if r['mode']=='incumbent'}
    seeds=json.loads((HERE/'panels.json').read_text())['development'];controls={};cases=[];n=0
    for seed in seeds:
        co=run(model,seed,None);n+=1
        assert co['layout']==previous[seed]['layout'] and co['turns']==previous[seed]['turn']
        for t in ('20','60','100','200','299'):
            assert [r['cash'] for r in co['marks'][t]]==[r['cash'] for r in previous[seed]['marks'][t]],(seed,t)
        controls[seed]=co
        for seat in (0,1):
            if co['edits'][seat]:
                got=run(model,seed,seat);n+=1
                if got['edits'][seat][0]!=co['edits'][seat][0]:raise ValueError('pair first divergence')
                got['identical_without_edit']=False
            else:
                got=dict(co,seat=seat,identical_without_edit=True)
            cases.append(got)
        print(seed,[(r['seat'],len(r['edits'][r['seat']]),r['marks']['299'][r['seat']]['cash']) for r in cases[-2:]],flush=True)
        result=summary(controls,cases)
        write(OUTPUT,dict(finished_maps=len(controls),nominal_games=len(controls)*3,simulated_games=n,baseline_sha256=sha(ROOT/'submission.py'),
                         income_sha256=sha(HERE/'income_loop.py'),receipt_sha256=sha(receipts.MODEL),summary=result,controls=controls,candidates=cases,
                         method='Four predeclared development layouts, both seats. One frozen root per player, no forward search. Control shadow tracks when the mission first changes a move, then stops on that seat. Unchanged controls are identical deterministic games; paired controlled rolls re-execute every step when a move does change. Positive receipt followed by a hire is an association, not proof of incremental financing.'))


if __name__=='__main__':main()
