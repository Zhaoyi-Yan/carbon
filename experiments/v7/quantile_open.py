"""Check a delayed mixed-role opening against historical opponents and the locked action holdout."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
from distill_data import load_manifest,local,configure_engine
from quantile_fit import ROOT

def first(path):
    # Decode only the initial observed boards.
    with gzip.open(path,'rt') as f:s=f.read(196608)
    m=re.search(r'"steps"\s*:\s*\[',s)
    if m is None:raise ValueError('missing steps')
    s=s[m.end():].lstrip();out=[]
    for _ in range(3):
        frame,b=json.JSONDecoder().raw_decode(s);out.append(frame[0]['observation']['players']);s=s[b:].lstrip().lstrip(',').lstrip()
    return out

def signature(frames,s):
    one,two=frames[1],frames[2];units=two[s][2]
    basic=not one[s][2] and len(two[s][1])==2 and len(units)==2 and {w[2] for w in units.values()}=={'COLLECTOR','PLANTER'}
    if not basic:return False,False
    a,b=list(one[s][1].values());target=None;other=list(one[1-s][2].values())
    if len(other)==2 and all(w[2]=='COLLECTOR' for w in other):target=max(a,b)
    elif len(other)==2 and s==0:
        p=next((w[0] for w in other if w[2]=='PLANTER'),None)
        if p is not None:target=b if b//15==p//15 else a
    seen=next(w[0] for w in units.values() if w[2]=='PLANTER')
    return True,(target is None or target==seen)

def main(data):
    configure_engine();manifest,fold,ban=load_manifest(data/'distill-public')
    metadata=json.loads((ROOT/'analysis/v7/distill-public.json').read_text())['matches']
    names={r['id']:next((r['name_'+a] for a in ('a','b') if r['name_'+a]!='yan-QQ-v6'),'yan-QQ-v6') for r in metadata}
    cohort=defaultdict(list);lookup={}
    for r in manifest:
        if r['source']=='local':
            it=local(r['payload'],0)
            frames=[next(it)[1]['players'] for t in range(3)]
        else:frames=first(r['payload'])
        for negative in (False,True):
            s=1-r['seat'] if negative else r['seat'];basic,consistent=signature(frames,s)
            key=r['source']+'-'+r['id']+('-other' if negative else '-v6')
            name=('yan-v5' if r['source']=='local' else names.get(r['id'],'unknown')) if negative else 'yan-QQ-v6'
            lookup[key]=dict(signature=consistent,basic=basic,name=name,layout=r['group'],fold=fold[r['group']])
            cohort[r['source'],fold[r['group']],'other' if negative else 'v6'].append((basic,consistent))
    archive=json.loads((data/'receipts-cache/index.json').read_text())['games'];indexes={};byname=defaultdict(list);used=[]
    for row in archive:
        if row['layout'] in ban or row['layout'] in {r['group'] for r in manifest}:continue
        src=ROOT/row['source']
        if src not in indexes:
            indexes[src]={Path(r['file']).name:r for r in json.loads((src/'index.json').read_text()) if r.get('file')}
        rec=indexes[src].get(row['file'])
        if not rec:continue
        frames=first(src/'replays'/row['file']);used.append(row['layout'])
        for s,k in enumerate(rec['order']):
            name=rec['strategies'][k]['name'];byname[name].append(signature(frames,s))
    def metric(vals):return dict(n=len(vals),basic=sum(int(x) for x,y in vals),matched=sum(int(y) for x,y in vals))
    report=dict(cohorts={'/'.join(k):metric(v) for k,v in cohort.items()},archived_layouts=len(set(used)),
        history={k:metric(v) for k,v in sorted(byname.items())},all_history=metric([e for v in byname.values() for e in v]),
        holdout_matches=[dict(id=k,**v) for k,v in lookup.items() if v['fold']=='test' and v['signature']],
        rule='Public state at t1 shows no workers; at t2 exactly two, one per role, and observed planter base agrees with every forced assignment in frozen react opening. If neither reaction rule applies the assignment is unconstrained.',
        caveat='Historical archive is a convenience sample comprising both seats of each layout, not a future ranked-opponent prior.')
    (data/'quantile-cache/openings.json').write_text(json.dumps(lookup)+'\n')
    (ROOT/'analysis/v7/quantile-opening.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['cohorts'],report['all_history'],sorted([(metric(v)['matched'],k,len(v)) for k,v in byname.items()],reverse=True)[:12])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);main(p.parse_args().data)
