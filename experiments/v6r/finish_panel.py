"""Verify a complete frozen panel and archive its round-robin games in declared map order."""
import argparse
import json
from pathlib import Path

from report import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('freeze',type=Path);p.add_argument('parts',nargs='+',type=Path);p.add_argument('--result',type=Path,required=True);p.add_argument('--summary',type=Path,required=True)
    a=p.parse_args();frozen=json.loads(a.freeze.read_text());rows=[json.loads(line) for part in a.parts for line in part.read_text().splitlines()]
    report=summarize(frozen,rows)
    if not report['complete']:raise SystemExit('incomplete panel')
    rank={seed:i for i,seed in enumerate(frozen['seeds'])};sha=frozen['sha256']
    rows.sort(key=lambda r:(rank[r['seed']],r['players'][0]['sha256']!=sha))
    result=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows)
    for dest,text in ((a.result,result),(a.summary,json.dumps(report,indent=2)+'\n')):
        if dest.exists() and dest.read_text()!=text:raise ValueError(f'different panel already stored in {dest}')
        if not dest.exists():dest.write_text(text)
    print(report['wins'],report['losses'],report['wins_by_seat'],report['average'],report['passed'])

if __name__=='__main__':main()
