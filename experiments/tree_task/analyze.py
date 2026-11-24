"""Summarize matched tree-task differences and check candidate admission."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean
import sys


def compare(rows):
    results = defaultdict(list)
    exits = Counter()
    coverage = Counter()
    for row in rows:
        case = row['case']
        arms = row['arms']
        t,side,uid = (case[k] for k in ('turn','seat','unit'))
        c=arms['continue_task']
        exits[c['exit'] or 'unresolved']+=1
        audit=row['audit']
        plant=dict(row['first_actions'][side])
        plant.pop(uid,None)
        coverage['natural_hold']+=row['original_plants']
        coverage['exact_proposed']+=any(a==plant for _,a in audit.get('proposed',[]))
        coverage['exact_admitted']+=plant in audit.get('admitted',[])
        coverage['exact_finalist']+=any(audit['admitted'][int(i)]==plant for i in audit.get('final_scores',{}))
        coverage['any_hold_admitted']+=any(uid not in a for a in audit.get('admitted',[]))
        coverage['forced_planted']+=arms['plant_once']['target_id'] is not None
        assert all(a['end']==299 for a in arms.values())
        for name,left,right in (('first_step','original','plant_once'),
                                ('persistence','plant_once','continue_task'),
                                ('whole_task','original','continue_task')):
            a,b=arms[left],arms[right]
            before,after=[r['marks']['299'] for r in (a,b)]
            final={field:round(after['ledger']['net'][field]-before['ledger']['net'][field],3)
                   for field in ('net_tree','base','hire','plants','seizes','deaths')}
            result=dict(seed=case['seed'],seat=side,turn=t,category=case['category'],proxy=row['proxy'],
                changed_trajectory=a['trajectory']!=b['trajectory'],
                delta=round(b['margin']-a['margin'],2),target_net=round(after['target_net']-before['target_net'],3),
                cash18=round(b['marks'][str(t+18)]['cash_margin']-a['marks'][str(t+18)]['cash_margin'],2),
                target18=round(b['marks'][str(t+18)]['target_net']-a['marks'][str(t+18)]['target_net'],3),
                **final)
            results[name].append(result)
    tables={}
    for label,group in results.items():
        maps={r['seed'] for r in group}
        by_map={s:[r['delta'] for r in group if r['seed']==s] for s in sorted(maps)}
        tables[label]=dict(results=group,means={k:round(mean(r[k] for r in group),3)
            for k in ('delta','cash18','target_net','target18','net_tree','base','hire')},
            positive=sum(r['delta']>0 for r in group),negative=sum(r['delta']<0 for r in group),
            changed=sum(r['changed_trajectory'] for r in group),
            consistent_positive=sum(min(x)>0 for x in by_map.values()),
            consistent_negative=sum(max(x)<0 for x in by_map.values()),by_map=by_map)
    return dict(exits=exits,coverage=coverage,comparisons=tables)


if __name__=='__main__':
    out=Path(sys.argv[1]).resolve()
    with gzip.open(out/'branches.jsonl.gz','rt') as f:
        result=compare(list(map(json.loads,f)))
    (out/'attribution.json').write_text(json.dumps(result,indent=2)+'\n')
