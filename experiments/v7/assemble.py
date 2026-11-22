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
import late_hire
import postcargo
import rollout_fog
import receipt_motion
import receipt_tail
import rollout_cache
import shipment_warning
import small_batches
import harbormaster
import breakwater
import dual_policy

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


VERSIONS=('quay','tidepool','pilot','jetty','narrows','soundings','pilotage','fairlead','tidewatch')


def create(name):
    if name in ('harbormaster','breakwater','crosscurrent'):
        if name=='harbormaster':
            code=harbormaster.build(create('soundings'))
        elif name=='breakwater':
            code=breakwater.build((ROOT/'yan-v6.py').read_bytes())
        else:
            code=dual_policy.build((ROOT/'yan-v6.py').read_text(),create('harbormaster'))
        if len(code.encode())>=10_000_000:raise ValueError('single-file tournament size')
        compile(code,name,'exec')
        return code
    stage=VERSIONS.index(name)
    weights='critic-shipping.npz' if stage>=VERSIONS.index('soundings') else 'shipping.npz'
    source=(ROOT/'yan-v6.py').read_bytes()
    code=challenger.build(source,preferred='intercept',bias=1.16,asset=.56,weights=HERE/weights)
    if stage>=VERSIONS.index('tidepool'):
        for plugin in (breakout,reserve,depot_entry,opponent_view,flow_switch,knockout):
            code=plugin.build(code)
    if stage>=VERSIONS.index('pilot'):
        code=last_collector.build(code)
    if stage>=VERSIONS.index('jetty'):
        code=early_income.build(code)
    if stage>=VERSIONS.index('narrows'):
        code=netcash.build(distill_fast.build(code))
        code=code.replace('experiments/v6r.','experiments/v7.',1)
    if stage>=VERSIONS.index('soundings'):
        code=rollout.build(code)
    if stage>=VERSIONS.index('pilotage'):
        code=depot_corridor.build(master.build(code))
    if stage>=VERSIONS.index('fairlead'):
        code=rollout_fog.build(code,mirror=True)
        code=late_hire.build(postcargo.build(code))
    if stage>=VERSIONS.index('tidewatch'):
        code=receipt_tail.build(receipt_motion.build(code))
        code=shipment_warning.build(code,.3799)
        code=small_batches.build(rollout_cache.build(code))
    if len(code.encode())>=10_000_000:raise ValueError('single-file tournament size')
    compile(code,name,'exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('version',choices=(*VERSIONS,'harbormaster','breakwater','crosscurrent'));p.add_argument('output',type=Path)
    a=p.parse_args()
    data=create(a.version).encode()
    with a.output.open('xb') as f:f.write(data)
    print(a.version,len(data),hashlib.sha256(data).hexdigest())
