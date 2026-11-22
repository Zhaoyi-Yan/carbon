"""Carry each player's observed resource mask into its hypothetical moves."""
import inspect


def _v7_next_known(previous,players,reflect=False):
    result=previous.copy()
    for side,pl in enumerate(players):
        seen=[]
        for p,_,k in pl[2].values():
            if k=='COLLECTOR':seen.append(p)
            else:seen.extend(NB[p])
        seen.extend(p for p,_ in pl[3].values())
        if seen:
            if reflect:seen=[q for p in seen for q in CLS_MEMBERS[CLS[p]]]
            result[side,seen]=True
    return result


def _v7_visible_views(board,previous):
    x=_v5_pair(board['players'],board['step'],board['carbon'],previous)
    seen=board.get('_v7_seen')
    if seen is not None:x[:,1]=seen.reshape(2,15,15)
    return x


def build(code,mirror=False):
    root='    own, enemy = _v5_rotated_heads(x,opponent_x,players[me],players[op])'
    turn="    if not s2.players[0][2] or not s2.players[1][2]:"
    hook='def _turn(o, configuration, acts):'
    if any(code.count(v)!=1 for v in (root,hook,turn)):raise ValueError('expected fog tracking and embedded referee')
    code=code.replace(root,"    state['forecast_known']=np.stack((x[1],opponent_x[1]) if me==0 else (opponent_x[1],x[1])).reshape(2,225).astype(bool)\n"+root)
    helpers=inspect.getsource(_v7_next_known)+'\n\n'+inspect.getsource(_v7_visible_views)+'\n\n'
    code=code.replace(hook,helpers+hook).replace(turn,"    known=o.get('_v7_seen')\n    if known is None:known=_ST.get('v5_history',{}).get('forecast_known')\n    if known is not None:board['_v7_seen']=_v7_next_known(known,s2.players,"+repr(mirror)+")\n"+turn)
    for board,old in [('g','players'),('board','prev'),('o','b[3]')]:
        spot=f"_v5_pair({board}['players'],{board}['step'],{board}['carbon'],{old})"
        if code.count(spot)!=1:raise ValueError(spot)
        code=code.replace(spot,f'_v7_visible_views({board},{old})')
    return code
