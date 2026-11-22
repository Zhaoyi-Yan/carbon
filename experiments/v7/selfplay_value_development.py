"""Eight fixed-budget dev games against frozen v6, compared with archived Lifeline on the same seats."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
import sys
if os.environ.get('PYTHONHASHSEED')!='0':
    os.environ['PYTHONHASHSEED']='0';os.execv(sys.executable,[sys.executable,*sys.argv])
import argparse,json,statistics
from pathlib import Path
from development_duel import game,model,single_thread,configure_engine
import selfplay_value_data as d


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();a.work.mkdir(exist_ok=True,parents=True)
    single_thread();configure_engine()
    fact=json.loads((d.OUT/'selfplay-value-factual.json').read_text());prior=json.loads((d.OUT/'lifeline-development.json').read_text())
    assert fact['fixed_development_trial_eligible'] and fact['candidate_sha256']==d.sha(a.candidate)
    assert prior['sha256']==d.FROZEN and prior['opponent']==d.sha(d.ROOT/'yan-v6.py')
    m=model(a.candidate,'selfplay_value_trial',True);rival=model(d.ROOT/'yan-v6.py','selfplay_v6_rival',True)
    rows=[]
    for ref in prior['games']:
        seed,seat=ref['seed'],ref['seat'];cache=a.work/(fact['candidate_sha256'][:10]+f'-{seed}-{seat}.json')
        if cache.exists():r=json.loads(cache.read_text())
        else:
            r=game(seed,seat,m,rival);cache.write_text(json.dumps(r)+'\n')
        assert r['layout']==ref['layout']
        r.update(reference_margin=ref['margin'],delta=round(r['margin']-ref['margin'],2));rows.append(r)
        print(seed,seat,r['margin'],ref['margin'],r['delta'],flush=True)
    summ=dict(wins=sum(r['margin']>0 for r in rows),losses=sum(r['margin']<0 for r in rows),reference_wins=sum(r['reference_margin']>0 for r in rows),
              mean_margin=round(statistics.mean(r['margin'] for r in rows),3),mean_difference=round(statistics.mean(r['delta'] for r in rows),3),
              positive_differences=sum(r['delta']>0 for r in rows),gained_wins=sum(r['margin']>0>=r['reference_margin'] for r in rows),lost_wins=sum(r['reference_margin']>0>=r['margin'] for r in rows))
    (d.OUT/'selfplay-value-development.json').write_text(json.dumps(dict(candidate=fact['candidate_sha256'],reference=prior['sha256'],opponent=prior['opponent'],
        reference_log=d.sha(d.OUT/'lifeline-development.json'),design=prior['design'],games=rows,summary=summ),indent=1)+'\n')
    print(summ,flush=True)

if __name__=='__main__':main()
