"""Build each tournament file from verified v6, its fitted heads and the named policy changes."""
import argparse
import hashlib
from pathlib import Path

import breakout
import challenger
import depot_entry
import flow_switch
import early_income
import distill_fast
import netcash
import knockout
import last_collector
import opponent_view
import reserve
import rollout
import depot_corridor
import master

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def create(name):
    source=(ROOT/'yan-v6.py').read_bytes()
    code=challenger.build(source,preferred='intercept',bias=1.16,asset=.56,weights=HERE/('critic-shipping.npz' if name in ('soundings','pilotage') else 'shipping.npz'))
    if name=='quay':return code
    for plugin in (breakout,reserve,depot_entry,opponent_view,flow_switch,knockout):
        code=plugin.build(code)
    if name in ('pilot','jetty','narrows','soundings','pilotage'):code=last_collector.build(code)
    if name in ('jetty','narrows','soundings','pilotage'):code=early_income.build(code)
    if name in ('narrows','soundings','pilotage'):
        code=netcash.build(distill_fast.build(code))
        code=code.replace('experiments/v6r.','experiments/v7.',1)
    if name in ('soundings','pilotage'):code=rollout.build(code)
    if name=='pilotage':code=depot_corridor.build(master.build(code))
    if len(code.encode())>=10_000_000:raise ValueError('single-file tournament size')
    compile(code,name,'exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('version',choices=('quay','tidepool','pilot','jetty','narrows','soundings','pilotage'));p.add_argument('output',type=Path)
    a=p.parse_args()
    data=create(a.version).encode()
    with a.output.open('xb') as f:f.write(data)
    print(a.version,len(data),hashlib.sha256(data).hexdigest())
