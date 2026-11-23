"""Materialize eligible public Git blobs in small resumable verified batches."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main(out,cache):
    repo=out/'partial'
    def git(*args,**kw):
        return subprocess.check_output(['git','-C',str(repo),*args],**kw)
    rev=git('rev-parse','HEAD').decode().strip()
    index=git('show',rev+':index.json')
    tree={line.split('\t')[1]:line.split()[2] for line in git('ls-tree','-r',rev).decode().splitlines()}
    ids={p['id'] for p in json.loads((out/'board.json').read_text())['strategies'][:10]}
    target=out/'public';(target/'replays').mkdir(exist_ok=True,parents=True)
    # Use cached archives only when their exact Git object matches.
    local={}
    for path in cache.glob('v*-data/**/*.json.gz'):
        local.setdefault(path.name,[]).append(path)
    needed=[]
    def oid(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
    for r in json.loads(index):
        if not ids.intersection(s['id'] for s in r['strategies'].values()):continue
        name=r['file'];p=target/name;expected=tree[name]
        if p.is_file() and oid(p.read_bytes())==expected:continue
        for f in local.get(p.name,[]):
            b=f.read_bytes()
            if oid(b)==expected:
                p.write_bytes(b);break
        else:needed.append((expected,p))
    for i in range(0,len(needed),6):
        group=needed[i:i+6]
        for retry in range(4):
            try:
                for sha,p in group:
                    b=git('cat-file','blob',sha,timeout=60)
                    assert oid(b)==sha
                    p.write_bytes(b)
                break
            except (subprocess.SubprocessError,OSError):
                if retry==3:raise
                time.sleep(2**retry)
    (target/'revision.txt').write_text(rev+'\n')
    temp=target/'index.tmp';temp.write_bytes(index);temp.replace(target/'index.json')


if __name__=='__main__':main(*[Path(x).resolve() for x in sys.argv[1:]])
