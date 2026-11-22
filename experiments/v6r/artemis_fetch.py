"""Rebuild the Artemis cohort cache from pinned public GitHub blobs and indices."""
import argparse
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from artemis import ROOT, REVISION

REPO = 'JUZI1/carbon-league-replays'
MANIFEST = ROOT/'analysis/v6r-artemis-games.jsonl'


def download(revision, path, validate):
    urls = (f'https://raw.githubusercontent.com/{REPO}/{revision}/{path}',
            f'https://api.github.com/repos/{REPO}/contents/{path}?ref={revision}')
    for i, url in enumerate(urls):
        try:
            payload = subprocess.check_output(['curl','-fsSL','--retry','2','--max-time','25',url])
            if i:
                payload = b64decode(json.loads(payload)['content'])
            validate(payload)
            return payload
        except (ValueError, AssertionError, OSError, EOFError, subprocess.CalledProcessError):
            if i:
                raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    data=args.data
    (data/'replays').mkdir(exist_ok=True,parents=True)
    lines=[json.loads(line) for line in MANIFEST.read_text().splitlines()]
    expected={(row['match'],row['game']):row for row in lines}
    revisions=sorted({row['revision'] for row in lines})

    def index(sha):
        path=data/(sha+'-index.json')
        if not path.exists():
            path.write_bytes(download(sha,'index.json',json.loads))
        return sha, {(r['id'],r['game']):r for r in json.loads(path.read_bytes())}

    with ThreadPoolExecutor(args.jobs) as pool:
        indices=dict(pool.map(index,revisions))
    metas=[]
    for key,row in expected.items():
        sha=row['revision']
        metas.append(dict(indices[sha][key],revision=sha))

    def get_game(m):
        expected_sha=expected[m['id'],m['game']]['file_sha256']
        path=data/m['file']
        def validate(raw):
            if hashlib.sha256(raw).hexdigest()!=expected_sha:
                raise ValueError('Mismatch in archived gzip bytes')
            g=json.loads(gzip.decompress(raw))
            assert g['rewards']==[m['scores'][k] for k in m['order']]
            assert len(g['steps'])-1==m['steps']
        if path.exists():
            validate(path.read_bytes())
        else:
            body=download(m['revision'],m['file'],validate)
            temp=path.with_suffix('.part')
            temp.write_bytes(body)
            temp.replace(path)
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(get_game,metas))
    (data/'index.json').write_text(json.dumps(metas,ensure_ascii=False))
    board=data/'final-snapshot-board.json'
    if not board.exists():
        board.write_bytes(download(REVISION,'leaderboard.json',json.loads))


if __name__=='__main__':
    main()
