"""Reconcile the income earned during each uninterrupted ownership spell of a QQ tree."""
import argparse
import json
from pathlib import Path
from starvation_replay import read_replay,cash_flows


def ownership(game,side):
    frames=game['steps'];cfg=game['configuration'];rows=[];owners={};active={}
    for tick in range(1,len(frames)):
        previous=frames[tick-1][0]['observation']['players']
        obs=frames[tick][0]['observation'];current=obs['players']
        flows,_,_,changes=cash_flows(frames[tick-1],frames[tick],cfg,owners)
        for uid,row in list(active.items()):
            if uid not in current[side][3]:
                row['ends']=tick
                row['reason']='taken' if uid in current[1-side][3] else 'expired'
                del active[uid]
        for t,recipient,kind,uid,pos,age in changes:
            if recipient!=side:continue
            # Plant and seizure prices are constant across all purchases of one type by one seat this turn.
            key='plant_cost' if kind=='plants' else 'seize_cost'
            cost=flows[side][key]/flows[side][kind]
            row=dict(id=uid,cell=pos,begins=t,kind=kind,cost=round(cost,4),gross_income=0.,first_repaid=None,ends=None)
            rows.append(row);active[uid]=row
        for uid,row in active.items():
            assert uid in current[side][3]
            row['gross_income']+=obs['trees'][uid][1]
            if row['first_repaid'] is None and row['gross_income']>=row['cost']:
                row['first_repaid']=tick
    for row in rows:
        row['gross_income']=round(row['gross_income'],4)
        if row['ends'] is None:row.update(ends=len(frames)-1,reason='end_of_game')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('replay',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();g=read_replay(a.replay)
    result=dict(side=1,scope='Reported per-tree income during each unbroken ownership spell, net comparison against that purchase only. Collision receipts on the tree are accounted separately in the full ledger.',spells=ownership(g,1))
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print([(x['begins'],x['ends'],x['cost'],x['gross_income']) for x in result['spells'] if x['begins']<=88])
