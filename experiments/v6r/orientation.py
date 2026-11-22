"""Expose equivalent positions from one standard orientation while preserving referee seat order."""
import argparse
import inspect
from pathlib import Path


def _r6_reflect_board(obs):
    flip=_R6_REFLECTION
    players=[]
    for cash,bases,workers,trees in obs['players']:
        players.append([cash,{uid:flip[pos] for uid,pos in bases.items()},
                        {uid:[flip[pos],value,role] for uid,(pos,value,role) in workers.items()},
                        {uid:[flip[pos],age] for uid,(pos,age) in trees.items()}])
    return dict(players=players,trees=obs.get('trees',{}),step=obs['step'],player=obs['player'],
                carbon=[obs['carbon'][pos] for pos in flip],remainingOverageTime=obs.get('remainingOverageTime',0.))


def _r6_reflect_actions(action):
    replacement={'LEFT':'RIGHT','RIGHT':'LEFT'}
    return {uid:replacement.get(a,a) for uid,a in action.items()}


def build(code):
    entry='def agent(observation, configuration):'
    if code.count(entry)!=1:raise ValueError('unexpected entry')
    helpers='\n_R6_REFLECTION = tuple((pos//15)*15+14-pos%15 for pos in range(225))\n'
    code=code.replace(entry,helpers+inspect.getsource(_r6_reflect_board)+'\n'+inspect.getsource(_r6_reflect_actions)+'\n\n'+entry)
    anchor='        return _act(observation, configuration, t0)'
    if code.count(anchor)!=1:raise ValueError('unexpected size-15 decision')
    code=code.replace(anchor,"""        if observation['player']==0:
            action=_act(_r6_reflect_board(observation), configuration, t0)
            return _r6_reflect_actions(action)
"""+anchor)
    compile(code,'<canonical-coordinates>','exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(build(a.source.read_text()))
