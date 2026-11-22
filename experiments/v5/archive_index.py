"""Recover published top-20 match files from dated public replay export snapshots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import gzip
from http.client import IncompleteRead
import json
from pathlib import Path
import time
from urllib.request import Request,urlopen


def fetch(url):
    for attempt in range(5):
        try:
            with urlopen(Request(url,headers={'User-Agent':'carbon-replay-research'}),timeout=35) as response:return response.read()
        except (OSError,IncompleteRead,EOFError):
            if attempt==4:raise
            time.sleep(1+attempt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('events',type=Path,help='public GitHub repository events, containing pushed revisions')
    p.add_argument('out',type=Path);p.add_argument('existing',type=Path);p.add_argument('--download',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);(a.out/'replays').mkdir(exist_ok=True)
    here=Path(__file__).parent;rank={r['id'] for r in json.loads((here/'design.json').read_text())['ranking']}
    old={r['match'] for r in json.loads(a.existing.read_text())}
    events=json.loads(a.events.read_text());by_time={}
    for e in sorted(events,key=lambda e:e['created_at'],reverse=True):
        info=e['payload']
        if e['type']=='PushEvent' and info.get('ref')=='refs/heads/main':
            dt=datetime.fromisoformat(e['created_at'].replace('Z','+00:00'));by_time.setdefault(int(dt.timestamp())//(2*3600),info['head'])
    def read(sha):
        path=a.out/(sha+'.json')
        if not path.exists():path.write_bytes(fetch('https://raw.githubusercontent.com/JUZI1/carbon-league-replays/'+sha+'/index.json'))
        return sha,json.loads(path.read_text())
    games={};sources=[]
    with ThreadPoolExecutor(3) as pool:
        for rev,rows in pool.map(read,by_time.values()):
            added=0
            for g in rows:
                n=Path(g['file']).name.split('.')[0]
                if (n in old or n in games or g['steps']!=299 or any(v!='DONE' for v in g['statuses'].values())
                        or not any(v['id'] in rank for v in g['strategies'].values())):continue
                games[n]=(g,rev);added+=1
            sources.append(dict(snapshot=rev,additional_games=added))
    (a.out/'index.json').write_text(json.dumps([g for g,rev in games.values()],ensure_ascii=False))
    (a.out/'sources.json').write_text(json.dumps(dict(snapshots=sources,
        games={n:rev for n,(g,rev) in games.items()}),indent=2)+'\n')
    print('snapshots',len(sources),'extra eligible matches',len(games),flush=True)
    if not a.download:return
    def download(item):
        game,rev=item;target=a.out/game['file']
        if target.exists():return
        raw=fetch('https://raw.githubusercontent.com/JUZI1/carbon-league-replays/'+rev+'/'+game['file'])
        replay=json.loads(gzip.decompress(raw))
        if len(replay['steps'])!=300:raise ValueError(game['file'])
        temp=target.with_suffix('.partial');temp.write_bytes(raw);temp.replace(target)
    with ThreadPoolExecutor(5) as pool:
        for i,_ in enumerate(pool.map(download,games.values()),1):
            if i%100==0:print('downloaded',i,flush=True)


if __name__=='__main__':
    main()
