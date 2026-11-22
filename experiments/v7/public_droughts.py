"""Compare observed interruptions to cash flow on distinct archived public competition maps."""
import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path
from receipt_data import LAND


def episode(frames,side):
    longest=0;streak=0;first=None
    for turn in range(30,min(251,len(frames))):
        ps=frames[turn][0]['observation']['players'];us,foe=ps[side],ps[1-side]
        cargo=sum(w[1] for w in us[2].values() if w[2]=='COLLECTOR')
        bad=us[0]<30 and cargo>=90 and foe[0]-us[0]>=60
        streak=streak+1 if bad else 0
        if streak>longest:longest,first=streak,turn-streak+1
    if first is None:return dict(length=0)
    own,other=[frames[first][0]['observation']['players'][s] for s in (side,1-side)]
    totals=[0,0]
    for turn in range(first,first+longest):
        ps=frames[turn][0]['observation']['players'];group,op=ps[side],ps[1-side]
        for pos,load,job in group[2].values():
            if job!='COLLECTOR' or load<30:continue
            threat=set()
            for entry,enemy,role in op[2].values():
                if role=='COLLECTOR' and enemy<=load:threat.update(LAND[entry])
            totals[0]+=all(h in threat for h in group[1].values());totals[1]+=1
    return dict(start=first,length=longest,cash=own[0],enemy_cash=other[0],trees=len(own[3]),enemy_trees=len(other[3]),
                workers=len(own[2]),enemy_workers=len(other[2]),both_gates=totals[0],loaded_worker_turns=totals[1])


def summary(rows):
    def rate(condition):return round(sum(condition(r) for r in rows)/len(rows),4) if rows else None
    total=sum(r.get('loaded_worker_turns',0) for r in rows)
    return dict(seats=len(rows),won=rate(lambda r:r['won']),without_trees=rate(lambda r:r.get('trees')==0),
                outnumbered_by_two=rate(lambda r:r.get('enemy_workers',0)>=r.get('workers',0)+2),
                median_trees=statistics.median(r['trees'] for r in rows) if rows else None,
                median_opposing_trees=statistics.median(r['enemy_trees'] for r in rows) if rows else None,
                both_depots_threatened_loaded_worker_fraction=round(sum(r.get('both_gates',0) for r in rows)/total,4) if total else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('index',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();pool=json.loads(a.index.read_text())['games'];indices={};seen=set();rows=[];missing=0
    for r in pool:
        root=Path(r['source'])
        if root not in indices:
            path=root/'index.json'
            entries=json.loads(path.read_text()) if path.exists() else []
            indices[root]={Path(x.get('file','')).name:x for x in entries if isinstance(x,dict)}
        meta=indices[root].get(r['file'])
        if not meta or not isinstance(meta.get('statuses'),dict) or set(meta['statuses'].values())!={'DONE'}:
            missing+=1;continue
        if r['layout'] in seen:continue
        seen.add(r['layout'])
        g=json.loads(gzip.decompress((root/'replays'/r['file']).read_bytes()))
        for side,key in enumerate(meta['order']):
            measures=episode(g['steps'],side)
            foe=meta['order'][1-side]
            rows.append(dict(match=meta['id'],layout=r['layout'],seat=side,strategy=meta['strategies'][key]['name'],
                             opponent=meta['strategies'][foe]['name'],won=meta.get('winner')==key,**measures))
    flags=[r for r in rows if r['length']>=10];transient=[r for r in rows if 1<=r['length']<10]
    by_enemy=defaultdict(list)
    for r in rows:by_enemy[r['opponent']].append(r)
    by_name={k:dict(seats=len(v),sustained=sum(r['length']>=10 for r in v),long=sum(r['length']>=50 for r in v)) for k,v in by_enemy.items()}
    result=dict(definition='For each seat take its longest consecutive stretch over frames 30..250 with cash < 30, carried cargo >= 90 and treasury deficit >= 60. A tie goes to the first stretch. Features are measured at the first frame of the selected stretch. Only games with both seats officially DONE and distinct original layouts.',cohort='Existing public replay training archive; opponent and map mix reflect archival sampling.',layouts=len(seen),unusable_metadata_or_status=missing,
                seats=len(rows),at_least_50=sum(r['length']>=50 for r in rows),at_least_100=sum(r['length']>=100 for r in rows),sustained=summary(flags),
                transient=summary(transient),by_opponent=by_name,spells=flags)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(len(seen),result['sustained'],result['transient'],flush=True)


if __name__=='__main__':
    main()
