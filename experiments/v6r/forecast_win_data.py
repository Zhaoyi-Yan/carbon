"""Align official game outcomes and current cash with the legal, map-separated forecast rows."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def align(raw,cache,out):
    with np.load(cache) as z:
        group=z['G'];turn=z['T'];games=json.loads(str(z['games']))
    cash=np.empty(len(group),"f");target=np.empty(len(group),"f")
    terminal=[];score_cash_diff=0;draws=0
    for i,r in enumerate(games):
        packed=(raw/r['file']).read_bytes()
        if hashlib.sha256(packed).hexdigest()!=r['sha256']:raise ValueError(r['file'])
        data=json.loads(gzip.decompress(packed));f=data['steps']
        score=data['rewards'][0]-data['rewards'][1]
        own,opp=f[-1][0]['observation']['players']
        terminal.append([float(score),own[0]-opp[0]])
        score_cash_diff+=int(np.sign(score)!=np.sign(own[0]-opp[0]));draws+=int(score==0)
        mask=group==i
        treasury=[q[0]['observation']['players'][0][0]-q[0]['observation']['players'][1][0] for q in f]
        cash[mask]=np.asarray(treasury)[turn[mask]];target[mask]=.5+.5*np.sign(score)
    np.savez_compressed(out,cash=cash,target=target,terminal=np.asarray(terminal),games_hash=hashlib.sha256(json.dumps(games,sort_keys=True).encode()).hexdigest())
    print(len(games),len(group),"draws",draws,"score-vs-cash terminal sign differences",score_cash_diff)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw',type=Path);p.add_argument('cache',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();align(a.raw,a.cache,a.output)
