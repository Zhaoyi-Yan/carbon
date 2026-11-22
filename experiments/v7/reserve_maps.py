"""Derive the second block of unseen map seeds from the frozen baseline before any challenger plays them."""
import hashlib
import json
from pathlib import Path

PATH=Path(__file__).with_name('panels.json')
LABEL='yan-v7-reserved-maps-09-to-16-v1'


def extend(design):
    old={k:design[k][:8] for k in ('qualifying','confirmation')}
    taken={s for group in old.values() for row in group for s in row}|set(design['development'])
    n=0
    for r in range(9,17):
        for group,count in (('qualifying',8),('confirmation',4)):
            row=[]
            while len(row)<count:
                digest=hashlib.sha256(f"{LABEL}/{design['frozen_v6']}/{n}".encode()).digest();n+=1
                s=int.from_bytes(digest[:8],'big')%1_000_000_000
                if s in taken:continue
                taken.add(s);row.append(s)
            old[group].append(row)
    return dict(design,**old,extension=dict(rounds=[9,16],seed_label=LABEL))


if __name__=='__main__':
    design=json.loads(PATH.read_text());expanded=extend(design)
    for kind in ('qualifying','confirmation'):
        if len(design[kind]) not in (8,16) or expanded[kind][:len(design[kind])]!=design[kind]:
            raise ValueError('The previous panel registry must be preserved exactly')
    PATH.write_text(json.dumps(expanded,indent=2)+'\n')
