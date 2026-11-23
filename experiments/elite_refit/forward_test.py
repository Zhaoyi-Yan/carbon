"""Reserve and evaluate newly finished public matches after the training corpus freeze."""
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
from collect import fetch,write
from dataset import extract,init
from fit import assess,evaldata
from network import make,PARENT


def main(out):
    cutoff=json.loads((out/'freeze.json').read_text())['time']
    teachers=json.loads((out/'teachers.json').read_text())
    ids={p['id']:i for i,p in enumerate(teachers)}
    seen={r['layout'] for r in json.loads((out/'split.json').read_text())}
    picked={};errors=[]
    for p in teachers:
        try:matches=json.loads(fetch('/api/matches?strategy='+p['id']))['matches']
        except OSError as exc:errors.append(str(exc));continue
        time.sleep(2.1)
        for m in matches:
            if (m.get('finished') or 0)<=cutoff or not m.get('replay_available'):continue
            for g in m.get('result') or []:
                if g['statuses']!={'a':'DONE','b':'DONE'}:continue
                key=f"{m['id']}-{g['game']}"
                picked[key]=dict(g,id=m['id'],finished=m['finished'],strategies={s:dict(id=m[s]) for s in 'ab'})
    maps=[]
    init()
    for key,r in sorted(picked.items()):
        p=out/'official'/(key+'.html.gz')
        try:
            if not p.is_file():
                raw=fetch(f"/replay/{r['id']}/{r['game']}");p.write_bytes(raw if raw[:2]==b'\x1f\x8b' else gzip.compress(raw,1))
                time.sleep(2.1)
            x=extract((key,dict(row=r,path=str(p),source='prospective',sha256=hashlib.sha256(p.read_bytes()).hexdigest()),ids,out/'encoded'))
            if x['layout'] not in seen:maps.append(dict(x,fold='forward'));seen.add(x['layout'])
        except (ValueError,OSError) as exc:errors.append([key,str(exc)])
    report=dict(cutoff=cutoff,maps=maps,errors=errors,listed_games=len(picked))
    if maps:
        rows=evaldata(out,maps,'forward')
        # Candidate files already contain extended styles. Restore the selected 37-row layer after loading.
        dev='mps' if torch.backends.mps.is_available() else 'cpu'
        for name in ('yan-v7','top6','top6-plus'):
            text=PARENT.read_text() if name=='yan-v7' else (out/(name+'.py')).read_text()
            if name=='yan-v7':net=make(text,False)
            else:
                net=make(text,False,extended=True)
            report[name]=assess(net.to(dev),rows,dev)
    write(out/'forward-test.json',report)


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
