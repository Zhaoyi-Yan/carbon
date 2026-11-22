"""Cache completed unrelated public QQ-v6 games from the recorded match list, sequentially."""
import argparse
import gzip
import json
from pathlib import Path
import ssl
import time
from urllib.request import Request,urlopen

EXCLUDE={'03100a95e8647c51a4d642fe08d856dd','4063a34afedb6d4767a9118e1dee9e5d'}
V6='36316605750f36aa828458e72269c6dc'
def fetch(meta,out):
    out.mkdir(parents=True,exist_ok=True);records=[]
    for row in json.loads(meta.read_text())['matches']:
        if not row['finished'] or row['id'] in EXCLUDE or not row['replay_available']:continue
        key='a' if row['a']==V6 else 'b';assert row[key]==V6
        for game in row['result']:
            filename=f"{row['id']}-{game['game']}.json.gz";f=out/filename
            if f.exists():
                with gzip.open(f) as s:g=json.load(s)
            else:
                url=f"https://117.72.210.134/replay/{row['id']}/{game['game']}"
                req=Request(url,headers={'Accept-Encoding':'gzip'})
                try:
                    with urlopen(req,context=ssl._create_unverified_context(),timeout=12) as stream:data=stream.read()
                    if data[:2]==b'\x1f\x8b':data=gzip.decompress(data)
                    g=json.JSONDecoder().raw_decode(data.decode().split('window.kaggle =',1)[1].lstrip())[0]['environment']
                except (OSError,ValueError,IndexError) as e:
                    print('deferred public file',filename,str(e)[:130],flush=True)
                    if '429' in str(e):return
                    time.sleep(3);continue
                pending=f.with_suffix('.tmp')
                with gzip.open(pending,'wt',compresslevel=1) as s:json.dump(g,s,separators=(',',':'))
                pending.replace(f)
                time.sleep(2.5)
            assert g['rewards']==[game['scores'][k] for k in game['order']]
            records.append(dict(file=filename,match=row['id'],seat=game['order'].index(key),seed=g['configuration']['randomSeed']))
            (out/'index.json').write_text(json.dumps(records));print(len(records),filename,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('list',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();fetch(a.list,a.out)
