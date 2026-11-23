"""Run the unchanged opening arm on twelve new paired maps."""
import json,hashlib,sys,os,subprocess
from pathlib import Path
from team_batch import run_panel, RESULTS, Strategy, ROOT, audit, summarize, configure_engine, limit_math_threads
from teamplay import build

def main():
    out=Path(sys.argv[1]).resolve()
    old=json.loads((RESULTS/'team/protocol.json').read_text())
    base=ROOT/'yan-v7.py'
    opening=out/'yan-v8-opening.py'
    opening.write_text(build(base.read_text(),'opening'))
    players={a:Strategy(a,p,hashlib.sha256(p.read_bytes()).hexdigest()) for a,p in [('opening',opening),('yan-v7',base)]}
    assert all(p.sha256==old['sha256'][a] for a,p in players.items())
    maps=[int(hashlib.sha256(f'opening-retest-1123/{i}'.encode()).hexdigest()[:8],16)%10**9 for i in range(12)]
    protocol=dict(screening=maps,confirmation=[],sha256={a:p.sha256 for a,p in players.items()},gate='At least 8 wins per seat and positive average on this independent panel.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    limit_math_threads();configure_engine();os.environ['CARBON_DEBUG']='1'
    trip=out/'previous.json'
    trip.write_text(json.dumps(json.loads((RESULTS/'trip-results.json').read_text())['freshness']))
    subprocess.run([sys.executable,str(Path(__file__).with_name('freshness.py')),str(out),'--previous',str(trip),*[str(RESULTS/p) for p in ('team/freshness.json','option-layouts.json','payback-layouts.json')]],check=True)
    raw=run_panel(out,'screening',('opening',),players,protocol,3)['opening']
    rows=[audit(r,protocol,(75,125)) for r in raw]
    assert sorted((r['seed'],r['seat']) for r in rows)==sorted((s,i) for s in maps for i in range(2))
    (out/'opening.json').write_text(json.dumps(dict(summary=summarize(rows),games=rows),indent=2))

if __name__=='__main__':main()
