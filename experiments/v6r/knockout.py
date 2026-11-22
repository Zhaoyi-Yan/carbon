"""Stop hypothetical play when the same bankruptcy condition as the official referee occurs."""
import argparse
import inspect
from pathlib import Path


def _r6_outcome(players, floor):
    dead=[not workers and not trees and (not bases or cash<floor)
          for cash,bases,workers,trees in players]
    if not any(dead):return None
    return int(dead[1])-int(dead[0])


def build(code):
    def replace(old,new):
        nonlocal code
        if code.count(old)!=1:raise ValueError(old)
        code=code.replace(old,new)
    anchor='def _turn(o, configuration, acts):'
    replace(anchor,inspect.getsource(_r6_outcome)+'\n\n'+anchor)
    anchor='    if not _RX:\n        return _NBoard'
    replace(anchor,"    if o.get('_terminal_sign') is not None:return o\n"+anchor)
    anchor="    return {'players': s2.players, 'step': s2.step, 'carbon': s2.carbon, '_st': s2, '_sap_previous':o['players']}"
    replace(anchor,"""    board={'players':s2.players,'step':s2.step,'carbon':s2.carbon,'_st':s2,'_sap_previous':o['players']}
    if not s2.players[0][2] or not s2.players[1][2]:
        floor=min(_RX_CFG[1]['recCollectorCost'],_RX_CFG[1]['recPlanterCost'])
        sign=_r6_outcome(s2.players,floor)
        if sign is not None:board['_terminal_sign']=sign
    return board""")
    replace("        return b[2]['step'] > LAST_TURN","        return b[2]['step'] > LAST_TURN or b[2].get('_terminal_sign') is not None")
    anchor="        c = [float(pl[0]) for pl in b[2]['players']]"
    replace(anchor,"        if b[2].get('_terminal_sign') is not None:return 3.*b[2]['_terminal_sign']*(1-2*me)\n"+anchor)
    anchor="        if board['step']>LAST_TURN:\n            cash="
    replace(anchor,"        if board.get('_terminal_sign') is not None:\n            values.append(3.*board['_terminal_sign']*(1-2*me))\n        elif board['step']>LAST_TURN:\n            cash=")
    replace("        if not work and cash<min(config.get('recCollectorCost',30),config.get('recPlanterCost',30)) and remaining>1:","        if not work and not trees and cash<min(config.get('recCollectorCost',30),config.get('recPlanterCost',30)) and remaining>1:")
    compile(code,'<referee-outcomes>','exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    with a.output.open('x') as f:f.write(build(a.source.read_text()))
