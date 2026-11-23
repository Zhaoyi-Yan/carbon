"""Run local encoding, training, paired games and prospective evaluation to completion."""
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

from collect import write
HERE=Path(__file__).resolve().parent


def main(out):
    # Resume completed atomic stages after an interrupted invocation.
    inputs=json.loads((out/'sources-local.json').read_text())
    if not (out/'freeze.json').exists():
        write(out/'freeze.json',dict(time=time.time(),games=len(inputs),
              newest_finished=max(x['row']['finished'] for x in inputs.values())))
        with gzip.open(out/'source-manifest.jsonl.gz','wt') as f:
            for key,entry in sorted(inputs.items()):
                row=entry['row']
                f.write(json.dumps(dict(game=key,source=entry['source'],sha256=entry['sha256'],
                     **{k:row[k] for k in ('order','strategies','scores','steps','finished','statuses')}))+'\n')
    for script,done in [('dataset.py','split.json'),('fit.py','router.json'),
                        ('evaluate.py','arena.json'),('forward_test.py','forward-test.json')]:
        if not (out/done).is_file():
            subprocess.run([sys.executable,str(HERE/script),str(out)],check=True)


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
