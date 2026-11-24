"""Measure whether the auditable sequential decoder retains more of ft3's behavior."""
import json
from pathlib import Path
import sys

from pilot import Actor, STYLE, load_ft3, games, seeds, summary, torch


if __name__=='__main__':
    path=Path(sys.argv[1]);out=Path(sys.argv[2])
    torch.set_num_threads(2)
    teacher=load_ft3(path)
    raw=load_ft3(path,full=True)
    style=teacher._POLICY.sid(STYLE)
    raw._SID=style;raw._ROOT_TEMP=.4
    raw._search=lambda o,c,e,action,*args:action
    dev='mps' if torch.backends.mps.is_available() else 'cpu'
    net=Actor(teacher._POLICY,style).to(dev)
    maps=seeds('decoder-same-maps',3)
    data={}
    for name,avoid in (('independent',False),('sequential',True)):
        trials=[games(teacher,net,None,[m],.4,.4,dev,raw=(raw,seat),coordinated=avoid)
                for m in maps for seat in (0,1)]
        data[name]=summary(dict(games=[r for t in trials for r in t['games']],
                                seconds=sum(t['seconds'] for t in trials)))
    out.write_text(json.dumps(data,indent=2)+'\n')
