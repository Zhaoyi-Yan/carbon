"""Verify actual root entry and sampled finalist changes for the optional experimental file."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from advantage import build
from advantage_fit import pick
from advantage_data import configure_engine,manifest,mod,collect

def main(a):
    configure_engine();q=json.loads(a.model.read_text());changes=json.load(gzip.open(a.changes))
    target=changes[0];metric=json.loads(a.report.read_text())
    assert a.output_policy.read_text()==build(a.parent.read_text())
    meta,record=next((k,v) for k,v in manifest() if k['panel']==target['panel'] and k['seed']==target['seed'] and k['side']==target['side'])
    baseline=json.loads((a.cache/(meta['name']+'.json')).read_text());m=mod(a.output_policy)
    original=m._adv_joint;timing=[]
    def timed(*args):
        start=time.perf_counter();out=original(*args);timing.append((time.perf_counter()-start)*1000);return out
    m._adv_joint=timed;current=collect(m,meta,record);changed=0
    for old,new in zip(baseline['records'],current['records']):
        i=pick(old,np.array(q['coef']),q['score_scale'],q['weight'])
        assert old['t']==new['t'] and old['acts']==new['acts'] and i==new['winner']
        changed+=int(i!=old['winner'])
    assert changed and timing
    out=dict(policy_sha256=hashlib.sha256(a.output_policy.read_bytes()).hexdigest(),parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),
             confirmed_states=len(timing),observed_changes=changed,overhead_ms_p50_p95=np.percentile(timing,[50,95]).round(3).tolist(),
             assembled_bytes=a.output_policy.stat().st_size,recommended_for_qualification=metric['eligible_for_game_test'])
    a.report_out.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('parent','output_policy','cache','model','changes','report','report_out'):p.add_argument(k,type=Path)
    main(p.parse_args())
