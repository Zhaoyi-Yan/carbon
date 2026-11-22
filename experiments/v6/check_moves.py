"""Re-run a selected candidate through its public observation path at the two last-worker decisions."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'analysis/denial'))
from branch import run,configure_engine,limit_math_threads


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot',type=Path)
    p.add_argument('candidate',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args();configure_engine();limit_math_threads()
    cases=json.loads((ROOT/'analysis/v6-elimination.json').read_text())['cases']
    out=[]
    for c in cases:
        path=a.snapshot/'replays'/(c['match']+'-1.json.gz')
        replay=json.loads(gzip.decompress(path.read_bytes()))
        rows=run('v6-case-'+c['match'][:8],a.candidate.resolve(),replay['steps'],replay['configuration'],
                 c['seat'],{c['end']-1})
        for row in rows:
            row['match']=c['match']
            row['candidate_sha256']=hashlib.sha256(a.candidate.read_bytes()).hexdigest()
        out.extend(rows)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    for row in out:print(row['step'],row['original'],row['action'])

if __name__=='__main__':main()
