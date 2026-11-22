"""Measure severe low-cash windows across completed QQ-v6 matches with published summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen
import ssl

TARGET='36316605750f36aa828458e72269c6dc'
HOST='https://117.72.210.134'
CTX=ssl._create_unverified_context()


def load(url,path):
    if not path.exists():
        with urlopen(url,context=CTX,timeout=18) as f:data=f.read()
        json.loads(data)
        path.write_bytes(data)
        time.sleep(2.15)
    return json.loads(path.read_text())


def case(summary):
    meta=summary['metadata'];ourkey=next(k for k,v in meta['strategies'].items() if v['id']==TARGET)
    seat=meta['order'].index(ourkey);other=meta['order'][1-seat]
    flags=[];scored=[]
    for row in summary['samples']:
        t=row[0];us=row[1+seat];them=row[2-seat]
        # Each observation is a ten-turn checkpoint. Do not presume what happened between them.
        if 30<=t<=250:
            bad=us[0]<30 and us[4]>=90 and len(us)>4 and them[0]-us[0]>=60
            flags.append((t,bad))
            if bad:scored.append([t,round(us[0],2),round(us[4],2),us[3],round(them[0],2),them[3]])
    longest=count=0;last=None
    for t,bad in flags:
        count=count+1 if bad and last is not None and last==t-10 else int(bad)
        longest=max(longest,count);last=t
    return dict(match=meta['id'],game=meta['game'],opponent=meta['strategies'][other]['name'],seat=seat,
                victory=meta['winner']==ourkey,final_margin=round(meta['scores'][ourkey]-meta['scores'][other],2),
                low_cash_checkpoints=scored,longest_consecutive_checkpoints=longest)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('list',type=Path);p.add_argument('cache',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();a.cache.mkdir(exist_ok=True);games=[];missing=[]
    for m in json.loads(a.list.read_text())['matches']:
        for g in m.get('result') or []:
            n=g['game'];file=a.cache/f'{m["id"]}-{n}.json'
            try:d=load(f'{HOST}/api/summary/{m["id"]}/{n}',file)
            except (OSError,ValueError) as e:
                missing.append([m['id'],n,str(e)]);continue
            games.append(case(d))
    a.out.write_text(json.dumps(dict(matches=games,missing=missing,
                       cohort="All finished QQ-v6 games in frozen latest /api/matches snapshot.",
                       event="At least 2 consecutive ten-turn checkpoints in turns 30..250 with cash <30, cargo >=90 and cash deficit >=60.",
                       source_snapshot_sha256=hashlib.sha256(a.list.read_bytes()).hexdigest()),indent=2,ensure_ascii=False)+'\n')
    for kind in (True,False):
        group=[r for r in games if (r['longest_consecutive_checkpoints']>=2)==kind]
        print(kind,len(group),sum(x['victory'] for x in group),flush=True)


if __name__=='__main__':main()
