"""Measure cold-process source loading, NumPy import, both decoded NPZs and standard policy construction."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
CHILD="""
import base64,io,json,sys,time
from pathlib import Path
start=time.perf_counter()
src=Path(sys.argv[1]).read_text()
scope={}
exec(compile(src,'embedded-policy-weights','exec'),scope)
base=scope['_WEIGHTS'];part=time.perf_counter()
harbor=scope['_dual_harbor_weights'](base);decoded=time.perf_counter()
import numpy as np
def policy(data):
    with np.load(io.BytesIO(base64.b64decode(data)),allow_pickle=False) as z:
        p={k:z[k].astype(np.float32) for k in z.files if k not in ('snames','meta')}
        return p,[str(i) for i in z['snames']],int(z['meta'][0])
p1=policy(base);p2=policy(harbor)
assert p1[0]['emb'].shape==(23,96) and p2[0]['emb'].shape==(27,96)
print(json.dumps([part-start,decoded-part,time.perf_counter()-decoded]))
"""
def main(path):
    timings=[]
    for _ in range(5):
        start=time.perf_counter()
        proc=subprocess.run([sys.executable,'-I','-B','-c',CHILD,str(path)],check=True,stdout=subprocess.PIPE,text=True)
        timings.append(json.loads(proc.stdout)+[time.perf_counter()-start])
    columns=list(zip(*timings));med=[sorted(col)[len(col)//2] for col in columns]
    report=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=path.stat().st_size,
          seconds=['read_compile_numpy_import_decode_frozen','rebuild_harbor','load_both_standard_policy_objects','wall_including_fresh_process'],
          trials=[[round(x,5) for x in row] for row in timings],median=[round(x,5) for x in med],
          note='Five independent processes, no persistent bytecode, OS file caches shared. Includes parsing both policy objects; network convolution initialization not included.')
    (ROOT/'analysis/v7/dual-weights-startup.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fragment',type=Path);main(p.parse_args().fragment)
