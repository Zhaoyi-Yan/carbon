"""Audit search-budget result JSON and compute economy, CPU and outcome summaries."""
import argparse,hashlib,json,statistics
from pathlib import Path
from itertools import chain

def median(v):return round(statistics.median(v),3)

def summarize(j):
    wide,deep=j['calibration']['choices'];totals={};games=[]
    assert len(j['panel'])==len(j['results'])==6
    for r in j['results']:
        c=r['map'];seat=c['seat'];t0=c['start'];w,d=r['arms'][wide],r['arms'][deep];assert w['source']==d['source']
        own,enemy=w['source'][seat],w['source'][1-seat]
        vals={name:round(own[name]-enemy[name],3) for name in ('cash','cargo','trees','workers')}
        row={'match':c['match'],'original':c['original'],'fold':c['fold'],'start':t0,'seat':seat,'map_seed':c['map_seed'],
             'policy_seed':c['policy_seed'],'starting_difference':vals,'arms':{},'paired':{}}
        for plan,case in [(wide,w),(deep,d)]:
            assert case['from_turn']==t0 and case['seat']==seat and case['map_seed']==c['map_seed']
            assert len(case['costs'])==len(case['opponent_costs'])==case['through']-t0 and len(case['actions_sha256'])==64
            assert case['result']==(case['reward'][seat]>case['reward'][1-seat])-(case['reward'][seat]<case['reward'][1-seat])
            last=case['milestones'][str(case['through'])]
            assert round(last['seats'][seat]['cash']-last['seats'][1-seat]['cash'],2)==case['margin']
            for t,m in case['milestones'].items():
                events=[x for x in case['events'] if x[0]<=int(t)]
                for k,s in [('own_gross',seat),('opp_gross',1-seat)]:
                    assert abs(sum(x[4] for x in events if x[1]==x[2]==s)-m[k])<.05
            phase={}
            for h in (10,20):
                m=case['milestones'][str(min(t0+h,case['through']))]
                phase[str(h)]={k:m[k] for k in ('own_gross','opp_gross','net_gross','cash_margin')}
            row['arms'][plan]=dict(won=case['result'],through=case['through'],terminal_margin=case['margin'],windows=phase,
                cpu_ms=median([x[0] for x in case['costs']]),simulated_transitions=median([x[1] for x in case['costs']]))
        for h in ('10','20'):
            row['paired'][h]={k:round(row['arms'][wide]['windows'][h][k]-row['arms'][deep]['windows'][h][k],3) for k in ('own_gross','opp_gross','net_gross','cash_margin')}
        row['paired']['end_margin']=round(w['margin']-d['margin'],2);games.append(row)
    for fold in ('dev','holdout'):
        rr=[r for r in j['results'] if r['map']['fold']==fold];rows=[r for r in games if r['fold']==fold];total={}
        for plan in (wide,deep):
            costs=list(chain.from_iterable(r['arms'][plan]['costs'] for r in rr));wins=[r['arms'][plan]['result'] for r in rr]
            total[plan]={'wins':wins.count(1),'ties':wins.count(0),'losses':wins.count(-1),'policy_decisions':len(costs),
                'cpu_ms':median([r[0] for r in costs]),'simulated_transitions':median([r[1] for r in costs]),
                'root_actions':median([r[2] for r in costs]),'effective_depth':median([r[3] for r in costs]),
                'mean_cpu_ms':round(statistics.mean(r[0] for r in costs),3)}
        total['wide_to_deep_ratio']={k:round(total[wide][k]/total[deep][k],4) for k in ('cpu_ms','mean_cpu_ms','simulated_transitions')}
        total['wide_minus_deep']={'avg_end_margin':round(statistics.mean(r['paired']['end_margin'] for r in rows),2),
              'positive_margins':sum(r['paired']['end_margin']>0 for r in rows),
              'negative_margins':sum(r['paired']['end_margin']<0 for r in rows),
              '10_net_gross':round(statistics.mean(r['paired']['10']['net_gross'] for r in rows),2),
              '20_net_gross':round(statistics.mean(r['paired']['20']['net_gross'] for r in rows),2)}
        totals[fold]=total
    return {'plans':[wide,deep],'calibration':j['calibration']['median'],'games':games,'summary':totals}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    report=summarize(json.loads(a.result.read_text()));report['raw_sha256']=hashlib.sha256(a.result.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(report,indent=1)+'\n');print(report['summary'])
