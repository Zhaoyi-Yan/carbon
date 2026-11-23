"""Freeze exact leaderboard IDs and the union of public archives and per-strategy recent games."""
import gzip
import hashlib
import json
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def fetch(url):
    # The league runs on its published IP with a self-signed certificate.
    request = urllib.request.Request('https://117.72.210.134'+url)
    for wait in (0, 5, 20):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(request,context=ssl._create_unverified_context(),timeout=30) as f:
                return f.read()
        except OSError:
            if wait == 20:
                raise


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main(out, cache):
    out=out.resolve()
    top=json.loads((out/'board.json').read_text())['strategies'][:10]
    ids={x['id'] for x in top}
    write(out/'teachers.json',[{'rank':i+1, **{k:x[k] for k in ('id','name','author')}} for i,x in enumerate(top)])
    files={}
    for parent in [out/'public',*sorted(cache.glob('v*-data/*')),*sorted(cache.glob('v*-data'))]:
        for manifest in ('index.json','snapshot-index.json'):
            index=parent/manifest
            if not index.is_file():continue
            rows=json.loads(index.read_text())
            if not isinstance(rows,list):continue
            for r in rows:
                if 'strategies' not in r or not ids.intersection(x['id'] for x in r['strategies'].values()):continue
                key=f"{r['id']}-{r['game']}"
                for p in (parent/r['file'],parent/Path(r['file']).name):
                    if p.is_file() and key not in files:
                        files[key]=dict(row=r,path=str(p),source='archive')
    local=cache/'leaderboard-20261123'
    official=out/'official'
    official.mkdir(exist_ok=True)
    errors=[]
    for teacher in top:
        listing=official/(teacher['id']+'.json')
        if not listing.exists():
            listing.write_bytes(fetch('/api/matches?strategy='+teacher['id']))
            time.sleep(2.1)
        for match in json.loads(listing.read_text())['matches']:
            if not match.get('replay_available') or not isinstance(match.get('result'), list):continue
            for result in match['result']:
                key=f"{match['id']}-{result['game']}"
                if key in files:continue
                p=local/(key+'.html.gz')
                if not p.exists():p=official/p.name
                if not p.exists():
                    try:
                        raw=fetch(f"/replay/{match['id']}/{result['game']}")
                        raw=gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw
                        if b'window.kaggle =' not in raw:raise ValueError('missing replay')
                        p.write_bytes(gzip.compress(raw,1))
                    except (ValueError,OSError) as e:
                        errors.append([key,str(e)])
                        continue
                    finally:time.sleep(2.1)
                r=dict(result,id=match['id'],finished=match['finished'],
                       strategies={s:dict(id=match[s],name=match['name_'+s]) for s in 'ab'})
                files[key]=dict(row=r,path=str(p.resolve()),source='official')
    good={k:v for k,v in files.items() if v['row']['statuses']=={'a':'DONE','b':'DONE'} and v['row']['steps']>0}
    for v in good.values():
        v['sha256']=hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()
    write(out/'sources-local.json',good)
    write(out/'coverage.json',dict(archive_commit=((out/'public/revision.txt').read_text().strip() if (out/'public/revision.txt').is_file() else subprocess.check_output(
        ['git','-C',str(out/'public'),'rev-parse','HEAD'],text=True).strip()),unavailable=errors,
        games=len(good),by_teacher={p['id']:sum(any(s['id']==p['id'] for s in v['row']['strategies'].values())
                                                   for v in good.values()) for p in top}))


if __name__=='__main__':
    main(*map(Path,sys.argv[1:]))
