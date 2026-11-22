"""Evaluate positive-yield extra stays against cheap frozen v6 on four paired development layouts."""
import json
import os
import sys
from statistics import mean
from opening_probe import HERE,OUT,ROOT,load_policy,sha
from benchmark import configure_engine,limit_math_threads
from branch_replay import write
from income_loop_probe import run
import yield_mine

MARKS=(20,40,60,75,100,155,200,299)


def analyze(controls,records):
    rows=[]
    for c in records:
        s=c['seat'];control=controls[c['seed']]
        deltas={}
        for t in MARKS:
            m=c['marks'][str(min(t,c['turns']))];o=control['marks'][str(min(t,control['turns']))]
            delta={k:round(m[s][k]-o[s][k],2) for k in ('cash','harvest','inbank','trees','workers','recruit_count','tree_income','cargo')}
            delta['margin']=round(m[s]['cash']-m[1-s]['cash']-o[s]['cash']+o[1-s]['cash'],2)
            deltas[str(t)]=delta
        last=c['marks'][str(c['turns'])]
        rows.append(dict(seed=c['seed'],seat=s,separate_game=not c['identical_without_edit'],stays=len(c['edits'][s]),
                         final_margin=round(last[s]['reward']-last[1-s]['reward'],2),delta=deltas))
    return dict(rows=rows,wins_by_seat=[sum(r['seat']==s and r['final_margin']>0 for r in rows) for s in (0,1)],
                improved=sum(r['delta']['299']['margin']>0 for r in rows),mean_delta={str(t):{k:round(mean(r['delta'][str(t)][k] for r in rows),2)
                    for k in rows[0]['delta'][str(t)]} for t in MARKS})


def main():
    os.environ['CARBON_DEBUG']='1';limit_math_threads();configure_engine();m=load_policy()
    yield_mine.move=m.move;yield_mine.WIDX=m.WIDX;yield_mine._V5_DIST=m._V5_DIST;yield_mine._v5_home=m._v5_home
    search=m._search
    def capture(obs,cfg,est,base,lw,*args):
        m.mine_est=est;m.mine_logits=lw
        return search(obs,cfg,est,base,lw,*args)
    m._search=capture
    def choose(obs,cfg,est,base,mem):
        return yield_mine._v7_yield_mine(obs,cfg,est,base,m.mine_logits,mem)
    refs={r['seed']:r for r in json.loads((OUT/'opening-probe.json').read_text())['records'] if r['mode']=='incumbent'}
    fresh='--replication' in sys.argv
    if fresh:
        from yield_mine_replication import freeze
        manifest=freeze();seeds=manifest['seeds']
    else:seeds=json.loads((HERE/'panels.json').read_text())['development']
    output=OUT/('yield-mine-replication.json' if fresh else 'yield-mine-probe.json')
    controls={};cases=[];games=0
    for seed in seeds:
        control=run(m,seed,None,choose=choose,inspect_gate=None,marks_at=MARKS);controls[seed]=control;games+=1
        if fresh:
            assert control['layout']==manifest['layouts'][str(seed)]['exact']
        else:
            for t in (20,60,100,200,299):
                assert [p['cash'] for p in control['marks'][str(t)]]==[p['cash'] for p in refs[seed]['marks'][str(t)]]
        for s in (0,1):
            if control['edits'][s]:
                c=run(m,seed,s,choose=choose,inspect_gate=None,marks_at=MARKS);c['identical_without_edit']=False;games+=1
                assert c['edits'][s][0]==control['edits'][s][0]
            else:c=dict(control,seat=s,identical_without_edit=True)
            cases.append(c)
        result=analyze(controls,cases)
        write(output,dict(cohort='independent_development' if fresh else 'initial_development',finished_maps=len(controls),simulated_games=games,policy_sha256=sha(HERE/'yield_mine.py'),
            frozen_sha256=sha(ROOT/'yan-v6.py'),method='Both sides recompute fixed, no-search v6 root actions each turn. One profitable extra stay per turn at most, steps 4 to 74. Interventions only rerun games with a first differing action. Development maps; no qualifying layouts.',
            summary=result,controls=controls,candidates=cases))
        print(seed,[(x['stays'],x['delta']['75']['harvest'],x['delta']['75']['margin'],x['delta']['299']['margin']) for x in result['rows'][-2:]],flush=True)


if __name__=='__main__':main()
