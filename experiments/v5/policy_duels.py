"""Measure the acting policies without multi-turn search on either side; diagnostic only."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from backbone import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--seeds',nargs='+',type=int,required=True)
    a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='carbon-policy-ablation-') as tmp:
        paths=[]
        for name,src in ((a.candidate.stem,a.candidate),('v4',ROOT/'opponents/yan-v4.py')):
            data=src.read_bytes()
            dest=Path(tmp)/(name+'.py')
            dest.write_bytes(data+b'\n\ndef _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):\n    return act0\n')
            paths.append(name+'-policy='+str(dest))
            print(name,hashlib.sha256(data).hexdigest(),flush=True)
        subprocess.run([sys.executable,str(ROOT/'arena.py'),'--strategies',*paths,'--seeds',*map(str,a.seeds),'--jobs','1',
                        '--output',str(a.output)],check=True)


if __name__=='__main__':
    main()
