"""Check the public mining and collision proxies against fully recorded referee ledgers."""
import json
import sys
from pathlib import Path
import numpy as np
import flow_switch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark import configure_engine, new_game, limit_math_threads


def main():
    configure_engine();limit_math_threads()
    from zerosum_env.envs.carbon import carbon
    def ranks(values):
        uniq,indices,counts=np.unique(values,return_inverse=True,return_counts=True)
        return (np.cumsum(counts)-.5*counts)[indices]
    y,x=np.indices((15,15)).reshape(2,-1)
    dr=abs(y[:,None]-y);dc=abs(x[:,None]-x)
    flow_switch._V5_DIST=np.minimum(dr,15-dr)+np.minimum(dc,15-dc)
    flow_switch.LAST_TURN=298;flow_switch.np=np
    result=[];order={'02':'canopy','03':'zenith','04':'relay'}
    for number,name in order.items():
        rows=json.loads((ROOT/'analysis'/f'v6r-qual-{number}-economy.json').read_text())['details']
        reference={(g['seed'],g['seat']):g['phases'] for g in rows}
        for line in (ROOT/'analysis'/f'v6r-qual-{number}-{name}.jsonl').open():
            g=json.loads(line);seat=int(g['players'][1]['name']==name)
            state,env=new_game(g['seed'],g['episode_steps']);hist={};ph=np.zeros((4,4));n=np.zeros(4)
            for t,(orders,expected) in enumerate(g['actions_and_post_cash']):
                players=state[0].observation.players
                if t==0:flow_switch._r6_economic_mode(hist,players,seat,0)
                before=np.array([hist['mine_gap'],hist['loot_gap']])
                for side,a in enumerate(orders):state[side].action=a
                carbon.compute_next_board(state,env)
                curr=state[0].observation.players
                if not np.allclose([p[0] for p in curr],expected,atol=.005,rtol=0):raise ValueError('cash')
                mode=flow_switch._r6_economic_mode(hist,curr,seat,t+1)
                new=np.array([hist['mine_gap'],hist['loot_gap']])
                ph[t//75,:2]+=new-.958*before
                ph[t//75,2]+=mode
                ph[t//75,3]+=mode>.5
                n[t//75]+=1
            for i,(data,ref) in enumerate(zip(ph,reference[(g['seed'],seat)])):
                result.append(dict(round=number,seed=g['seed'],seat=seat,period=i,
                      proxy_mine=-data[0],true_mine=ref['harvest'],proxy_loot=data[1],true_loot=ref['enemy_loot'],
                      influence=data[2]/n[i],strong_switch=data[3]/n[i],next_bank=None if i==3 else reference[(g['seed'],seat)][i+1]['net_deposits']))
    metrics={}
    for k in ('mine','loot'):
        a=np.array([r['proxy_'+k] for r in result]);b=np.array([r['true_'+k] for r in result])
        metrics[k]=dict(spearman=float(np.corrcoef(ranks(a),ranks(b))[0,1]),mae=float(np.abs(a-b).mean()),mean_true_abs=float(np.abs(b).mean()))
    report=dict(window_count=len(result),maps=len(set(r['seed'] for r in result)),action_turns=48*299,metrics=metrics,rows=result)
    path=ROOT/'experiments/v6r/flow-check.json'
    with path.open('x') as f:json.dump(report,f,indent=2)
    print(metrics)


if __name__=='__main__':main()
