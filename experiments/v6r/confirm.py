"""Evaluate an independently reserved four-map same-file confirmation."""
import argparse
import json
from pathlib import Path

from freshness import check
from report import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('freeze', type=Path)
    parser.add_argument('results', type=Path)
    parser.add_argument('qualification', type=Path, help='completed eight-map report')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.freeze.read_text())
    original = json.loads(args.qualification.read_text())
    rows = [json.loads(line) for line in args.results.read_text().splitlines()]
    if not original['passed'] or original['sha256'] != frozen['sha256'] or len(frozen['seeds']) != 4:
        parser.error('confirmation requires a qualified matching file and four reserved maps')
    if set(frozen['seeds']) & {row[0] for row in original['games']}:
        parser.error('confirmation maps overlap the original panel')
    result = summarize(frozen, rows)
    result['freshness'] = check(frozen, rows)
    result['passed'] = bool(result['complete'] and not result['invalid'] and
                            result['wins'] >= 5 and min(result['wins_by_seat']) >= 2 and result['average'] > 0)
    result['criterion'] = frozen['criterion']
    if not result['complete']:
        parser.error('incomplete confirmation')
    combined = [x for row in original['games']+result['games'] for x in row[1:]]
    if all(x is not None for x in combined):
        result['both_panels'] = dict(maps=len(combined)//2, wins=sum(x>0 for x in combined), losses=sum(x<0 for x in combined),
            wins_by_seat=[a+b for a,b in zip(original['wins_by_seat'], result['wins_by_seat'])],
            average=round(sum(combined)/len(combined),2))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['wins'],result['wins_by_seat'],result['average'],result['passed'])


if __name__=='__main__':
    main()
