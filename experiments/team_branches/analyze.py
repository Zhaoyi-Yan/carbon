"""Summarize matched asset and cash deltas by root, horizon and search stage."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean
import sys


def search_stage(hit):
    if hit['selected']:
        return 'selected'
    if hit['finalist_score'] is not None:
        return ('ranked_below' if hit['finalist_score']<hit['winning_score'] else 'finalist_tie')
    if hit['admitted']:
        return 'pruned'
    if hit['proposed']:
        return 'generated_not_admitted'
    return 'not_generated'


def flags(values):
    return dict(better=sum(v>0 for v in values),worse=sum(v<0 for v in values),
                tied=sum(v==0 for v in values),mean=round(mean(values),2))


def compare(rows):
    grouped=defaultdict(list)
    for row in rows:
        c=row['case']
        unchanged=row['arms']['original']
        t=c['turn']
        for name,arm in row['arms'].items():
            if name=='original':
                continue
            rec=dict(seed=c['seed'],side=c['seat'],turn=t,opponent=c['opponent'],response=row['response'],
                     root=c['match'],same_reply=row['same_reply'],ready=arm['ready'],
                     differs=arm['played']['trace']!=unchanged['played']['trace'],
                     stage=search_stage(arm['audit']),audit=arm['audit'],
                     altered_after_search=arm.get('altered_after_search'))
            for tag,turn in (('10',t+10),('20',t+20),('terminal',299)):
                a=unchanged['played']['marks'][str(turn)]
                b=arm['played']['marks'][str(turn)]
                cash=[round(y-x,2) for x,y in zip(a['team_cash_change'],b['team_cash_change'],strict=True)]
                ledger={k:round(b['ledger']['net'][k]-a['ledger']['net'][k],2)
                        for k in ('base','net_tree','hire','plants','seizes','deaths')}
                rec[tag]=dict(cash=round(b['net_cash']-a['net_cash'],2), own_cash=cash[0],
                              enemy_cash=cash[1], **ledger)
            grouped[c['cohort'],name].append(rec)
    report={}
    for (cohort,name),rr in grouped.items():
        # Each root contributes equal weight. Repeated first replies count once.
        distinct=[r for r in rr if not (r['same_reply'] and r['response']=='historical')]
        by_root=defaultdict(list)
        for r in distinct:
            by_root[r['root']].append(r)
        root_means={tag:[mean(r[tag]['cash'] for r in vals) for vals in by_root.values()]
                    for tag in ('10','20','terminal')}
        ready=[r for r in distinct if r['ready'] and r['differs']]
        stage=Counter(r['stage'] for vals in by_root.values() for r in vals[:1] if r['ready'])
        steady=defaultdict(Counter)
        for vals in by_root.values():
            if len(vals)!=2 or not all(r['ready'] and r['differs'] for r in vals):
                continue
            for tag in ('10','20','terminal'):
                if all(r[tag]['cash']>0 for r in vals):
                    steady[tag][vals[0]['stage']]+=1
        ledger={}
        for tag in root_means:
            fields=('own_cash','enemy_cash','base','net_tree','hire')
            ledger[tag]={k:round(mean(mean(r[tag][k] for r in v) for v in by_root.values()),2) for k in fields}
        report.setdefault(cohort,{})[name]=dict(
            roots=len(by_root),comparisons=len(distinct),ready_changes=len(ready),
            means={tag:flags(vals) for tag,vals in root_means.items()},ledger_means=ledger,search_stages=stage,
            strictly_positive_both_replies=steady,rows=rr)
    return report


if __name__=='__main__':
    out=Path(sys.argv[1]).resolve()
    with gzip.open(out/'branches.jsonl.gz','rt') as f:
        rows=list(map(json.loads,f))
    assert len(rows)==24
    assert all(a['played']['end']==299 for row in rows for a in row['arms'].values())
    summary=compare(rows)
    (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
