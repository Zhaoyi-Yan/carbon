"""How different and credible is the optional sampled control after turn 190?"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from collections import Counter,defaultdict
from pathlib import Path
import argparse,gzip,hashlib,json
from surrogate import STYLE, ROOT, _engine, _joint, make
from surrogate_check import TARGET_LAYOUT,TARGET_SEED,metrics,score
from surrogate_variation import TEMPERATURE,perturb


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();d=a.recent
    source=json.loads((d/'index.json').read_text())
    m=_engine();totals=defaultdict(Counter);different=Counter();detail=[]
    trained=json.loads((ROOT/'experiments/v6r/elite-validation.json').read_text())
    excluded={q.split('-')[0] for q in trained['training']+trained['validation']}
    for row in source['rows']:
        path=d/row['file']
        if not path.is_file():continue
        assert row['id'] not in excluded
        g=json.load(gzip.open(path));cfg=g['configuration']
        if cfg['randomSeed']==TARGET_SEED:continue
        frames=g['steps'];p0=frames[0][0]['observation']
        board=[p0['full_carbon'],[sorted(p[1].values()) for p in p0['players']]]
        assert TARGET_LAYOUT not in {hashlib.sha256(json.dumps(o,separators=(',',':')).encode()).hexdigest()
                                     for o in (board,[cfg['size']]+board)}
        seat=next(i for i,k in enumerate(row['order']) if row['strategies'][k]['id']==STYLE['artemis'][0])
        art=make('artemis',cfg['randomSeed'],cfg);ap=make('apollo',cfg['randomSeed'],cfg)
        local=Counter()
        for t,fr in enumerate(frames[:-1]):
            o=fr[0]['observation'];pl=o['players'];truth=frames[t+1][seat]['action'] or {}
            obs=dict(step=t,player=seat,players=pl,trees=o['trees'],carbon=fr[seat]['observation']['carbon'])
            if t<190:
                art.observe(obs,truth);ap.observe(obs,truth);continue
            expect=art.predict(obs);art._finish(obs,truth)
            est,known=ap._step(obs);w,b=ap.logits(obs,est,known)
            clean=_joint(pl,seat,w,b,m,cfg)
            w,b=perturb(w,b,cfg['randomSeed'],t,seat)
            mix=_joint(pl,seat,w,b,m,cfg);ap._finish(obs,truth)
            for name,actions in [('artemis',expect),('apollo',clean),('apollo-varied',mix)]:
                totals[name].update(score(m,pl,seat,truth,actions,cfg))
            for cid in pl[seat][2]:
                local['workers']+=1
                local['two_embeddings_diff']+=expect.get(cid)!=clean.get(cid)
                local['varied_artemis_diff']+=expect.get(cid)!=mix.get(cid)
            local['joint_turns']+=1;local['varied_joint_diff']+=expect!=mix
        different.update(local)
        detail.append(dict(match=row['id'],seat=seat,seed=cfg['randomSeed'],difference=dict(local)))
    report=dict(temperature=TEMPERATURE,rule='i.i.d. Gumbel noise in raw logits, shared across branches by seed, step, seat, cell and action; no target data used.',
                source_revision=source['revision'],phase_start=190,by_game=detail,totals=dict(different),
                teacher_forced={k:metrics(v) for k,v in totals.items()})
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=1)+'\n');print(json.dumps(report['totals']))
    print(json.dumps(report['teacher_forced']['apollo-varied']))
if __name__=='__main__':main()
