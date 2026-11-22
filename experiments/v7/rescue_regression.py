"""Run a frozen one-engine release candidate against the calibrated Master7sE clone, paired with archived v6."""
import argparse
import gzip
import json
import os
from pathlib import Path
import torch
import master_e_branch as replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('file',type=Path);p.add_argument('sha');p.add_argument('output',type=Path)
    a=p.parse_args();replay.limit_math_threads();replay.configure_engine();os.environ['CARBON_DEBUG']='1'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    name=a.file.stem;agent=replay.load_ours(name,a.file,a.sha);clone=replay.Clone()
    raw=json.loads(gzip.decompress((replay.OUT/'branch-breakout-replays.json.gz').read_bytes()))['loss']
    assert raw['match_id']==replay.MATCH
    cfg=raw['configuration'];raw['actions']=raw['actions'][:164];raw['expected']=raw['expected'][:165]
    snaps=replay.history(raw,{name:agent},clone)
    prior=json.loads((replay.OUT/'master-e-rescue.json').read_text())
    assert prior['frozen_sha256']['v6']==replay.FROZEN['v6']
    rows=[]
    for start,end in replay.START_END:
        for seed in replay.SEEDS:
            base,=[r for r in prior['cases'] if (r['arm'],r['start'],r['rng_seed'])==('v6',start,seed)]
            row=replay.branch(cfg,snaps[start],clone,agent,name,start,end,seed)
            rows.append(dict(candidate=row,baseline_margin=base['final_cash_margin'],baseline_first=base['first'].get('own_seat1')))
            print(start,seed,row['first'].get('own_seat1'),row['final_cash_margin'],flush=True)
    a.output.write_text(json.dumps(dict(sha256=a.sha,baseline_sha=replay.FROZEN['v6'],match=replay.MATCH,seeds=replay.SEEDS,
        design='Same official positions, independently reacting previously calibrated E clone, width 4/depth 2/one reply/3-ply tail. Baseline compared using the byte-identical recorded v6 arm.',cases=rows),indent=2)+'\n')


if __name__=='__main__':main()
