"""Fetch only the nine independent validation replays from the pinned public snapshot."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

REV='3d96b8d032a102c297968e40911d893c9abc160d'
ROOT=Path(__file__).resolve().parents[2]
API='https://api.github.com/repos/JUZI1/carbon-league-replays/contents/'


def content(name):
    data=subprocess.check_output(['curl','-fsSL','--retry','2','--max-time','30',API+name+'?ref='+REV])
    return base64.b64decode(json.loads(data)['content'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    args=p.parse_args();d=args.data;(d/'replays').mkdir(parents=True,exist_ok=True)
    result=json.loads((ROOT/'analysis/v7/surrogate-validation.json').read_text())
    expected={r['match']:r for r in result['games'] if r['cohort']=='post_training'}
    rows=[r for r in json.loads(content('index.json')) if r['id'] in expected]
    assert len(rows)==len(expected)
    def fetch(row):
        target=d/row['file'];raw=target.read_bytes() if target.is_file() else content(row['file'])
        game=json.loads(gzip.decompress(raw))
        first=game['steps'][0][0]['observation']
        layout=hashlib.sha256(json.dumps([first['full_carbon'],[sorted(p[1].values()) for p in first['players']]],separators=(',',':')).encode()).hexdigest()
        assert layout==expected[row['id']]['layout'] and game['configuration']['randomSeed']==expected[row['id']]['seed']
        assert game['rewards']==[row['scores'][slot] for slot in row['order']]
        if not target.exists():target.write_bytes(raw)
    with ThreadPoolExecutor(3) as pool:list(pool.map(fetch,rows))
    (d/'index.json').write_text(json.dumps(dict(revision=REV,rows=rows)))
if __name__=='__main__':main()
