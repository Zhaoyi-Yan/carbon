"""Replay all changed paths, audit terminal censoring, and compare the shipped chooser with its offline scorer."""
import argparse,gzip,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from postcargo_check import fixed,replay_state,tail,run
from postcargo import build

def main(a):
    blob=build(a.parent.read_text());assert a.candidate.read_text()==blob
    reports=[json.loads(p.read_text()) for p in sorted(a.results.glob('*.json'))]
    manifest={r['match']:r for r in json.loads((a.public/'index.json').read_text())}
    m,stock=fixed(a.parent);calls=0;terminal=set();root_checks=0;smoke=0;m._POST_TIMES=([],[])
    start=blob.index('def _reply_choice(');stop=blob.index('    return best\n',start)+len('    return best\n')
    gl=dict(m.__dict__);exec(compile(blob[start:stop],'<actual shipped last chooser>','exec'),gl)
    # Preserve the production function's live module globals, including its real candidate samplers.
    import types
    m._POST_PRODUCTION=types.FunctionType(gl['_reply_choice'].__code__,m.__dict__)
    for r in reports:
        mr=manifest[r['match']]
        with gzip.open(a.public/mr['file']) as f:g=json.load(f)
        frames=g['steps'];cfg=g['configuration'];states,_=replay_state(m,g)
        for case in r['data']:
            if not case.get('outcomes'):continue
            t=case['t'];s=r['side'];before=case['searched_action'];dedup={}
            for mode,edits in case['changes'].items():
                first=dict(before)
                for e in edits:
                    if e['to_action']!='NONE':first[e['uid']]=e['to_action']
                    else:first.pop(e['uid'],None)
                tag=tuple(sorted(first.items()))
                if tag not in dedup:
                    original=m._turn
                    def audit(board,c,commands):
                        nxt=original(board,c,commands)
                        if any(not p[2] and not p[3] and (not p[1] or p[0]<min(c['recCollectorCost'],c['recPlanterCost'])) for p in nxt['players']):
                            terminal.add((r['match'],t,mode,nxt['step']))
                        return nxt
                    m._turn=audit
                    try:dedup[tag]=tail(m,states[t],t,s,cfg,frames,first)
                    finally:m._turn=original
                    calls+=1
                assert case['outcomes'][mode]==dedup[tag]
            root_checks+=1
        if smoke<2:
            # The entire sampled prefix of this game is required for the same frozen style history.
            turns=sorted(t['t'] for t in r['data'])
            saved=[q for q in r['data'] if q['changes']['corrected']!=q['changes']['legacy']]
            if saved:
                until=saved[0]['t'];subset=[t for t in turns if t<=until]
                cases,_=run(m,stock,g,r['side'],subset,False)
                assert any(t['changes']['corrected']!=t['changes']['legacy'] for t in cases)
                smoke+=1
    times=np.asarray(m._POST_TIMES)
    out=dict(candidate_sha256=hashlib.sha256(blob.encode()).hexdigest(),parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),
             corrected_and_original_chooser_equivalence_states=times.shape[1],verified_changed_states=root_checks,
             verified_distinct_ten_turn_command_paths=calls,premature_terminal_paths=[list(v) for v in sorted(terminal)],
             paired_timing_ms_p50_p95={name:np.percentile(v,[50,95]).round(3).tolist() for name,v in zip(('legacy','corrected','incremental'),(times[0],times[1],times[1]-times[0]))},
             note='Every re-evaluated edited path matched its saved metrics, with unchanged real rival command on the first ply. Integration clocks are measured on full enumerations; the ordinary agent retains its real-time cutoff.')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('candidate',type=Path);p.add_argument('results',type=Path);p.add_argument('public',type=Path);p.add_argument('output',type=Path)
    main(p.parse_args())
