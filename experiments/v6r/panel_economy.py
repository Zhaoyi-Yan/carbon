"""Reconcile each seed/seat in a recorded qualification against its actual final score."""
import argparse
import json
from pathlib import Path
from statistics import mean

from trace_economy import V5_SHA,check_near,configure_engine,contributions,limit_math_threads,replay_actions


def breakdown(record,seat):
    ledger=record['ledger'];cash=contributions(ledger['cash'],seat);other=1-seat
    def difference(section,field):
        table=ledger[section]
        return table[seat][field]-table[other][field]
    cash_diff=sum(cash.values())
    return dict(score_change=cash_diff,**cash,harvest=difference('cargo','harvest'),
                enemy_loot=difference('cargo','loot'),
                cargo_lost_to_collision=-difference('cargo','collision'),
                ground_tree_income=difference('tree_sources','ground'),
                tree_income_from_enemy_cargo=difference('tree_sources','opposing_cargo'),
                tree_turns=difference('counts','eligible_tree_turns'))


def run(files,sha,baseline_sha=V5_SHA):
    out=[]
    for filename in files:
        for line in filename.read_text().splitlines():
            result=json.loads(line);turns=result.pop('actions_and_post_cash')
            if result['status']!='completed':continue
            summary,_=replay_actions(result,turns,sha,baseline_sha);seat=summary['v6_seat']
            final=breakdown(summary,seat)
            check_near(final['score_change'],result['cash'][seat]-result['cash'][1-seat],'terminal cash decomposition')
            out.append(dict(seed=result['seed'],seat=seat,terminal_cash=result['cash'],result=final,
                            phases=[breakdown(p,seat) for p in summary['phases']]))
    if len({(g['seed'],g['seat']) for g in out})!=len(out):raise ValueError('duplicate result')
    def group(rows):return {k:round(mean(r[k] for r in rows),2) for k in rows[0]}
    return dict(candidate_sha256=sha,games=len(out),average=group([r['result'] for r in out]),
                first_75=group([r['phases'][0] for r in out]),details=out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('freeze',type=Path);p.add_argument('files',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();configure_engine();limit_math_threads()
    frozen=json.loads(a.freeze.read_text())
    data=run(a.files,frozen['sha256'],frozen.get('opponent_sha256',V5_SHA))
    a.output.write_text(json.dumps(data,indent=2)+'\n')
    print(data['games'],data['average'],flush=True)
