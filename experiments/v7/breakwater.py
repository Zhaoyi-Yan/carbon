"""Keep Estuary's full policy for ordinary boards and reopen its search during a depot race or drought."""
import inspect
import breakout
import challenger
import depot_corridor
import depot_entry
import early_income
import knockout
import last_collector
import late_hire
import receipt_motion
import receipt_tail
import reserve
import rollout_cache
import shipment_warning


def _v7_depot_race(root, side):
    us, them = root['players'][side], root['players'][1-side]
    poor = us[0] < 60 and len(us[3]) <= 2 and len(them[2]) >= len(us[2])+2
    cargo = sum(load for _,load,role in us[2].values() if role=='COLLECTOR')
    if poor and cargo >= 30:
        return 'recover'
    if root['step'] < 70 or LAST_TURN+1-root['step'] < 2:
        return None
    raw = _v7_flows(root)
    tail = _v7_receipt_tail(root)
    ours, enemy = raw[side,1], raw[1-side,1]
    if tail is not None and tail[1-side,0] >= _V7_LARGE_WARNING and ours >= 65:
        return 'contest'
    if root['step'] >= 190 and abs(us[0]-them[0]) < max(ours,enemy)+90 and abs(ours-enemy) >= 175:
        return 'contest'
    return None


def build(raw):
    v6=raw.decode()
    code=challenger.build(raw,preferred='intercept',bias=1.16)
    for plugin in (breakout,reserve,depot_entry,knockout,last_collector,early_income,depot_corridor,late_hire,receipt_motion,receipt_tail):
        code=plugin.build(code)
    code=shipment_warning.build(code,.3799)
    start=v6.index('def _r6_asset_score(board,side):')
    end=v6.index('\n_R6_CARGO = ',start)
    baseline=v6[start:end].replace('def _r6_asset_score(','def _v7_estuary_assets(',1)
    marker='def _r6_asset_score(board,side):'
    trigger='    root_modes={}\n    if most>=3:'
    reset='    _fc_track(obs)\n    st = _ST'
    for anchor in (marker,trigger,reset):
        if code.count(anchor)!=1:
            raise ValueError('Unrecognized Estuary hook: '+anchor)
    code=code.replace(reset,reset+"\n    st['v7_depot_race']=None")
    code=code.replace(marker,baseline+'\n\n'+inspect.getsource(_v7_depot_race)+'\n\n'+marker+
                      "\n    if not _ST.get('v7_depot_race'):return _v7_estuary_assets(board,side)")
    code=code.replace(trigger,"    _ST['v7_depot_race']=_v7_depot_race(root,me)\n    root_modes={}\n    if most>=3 and _ST['v7_depot_race']:")
    return rollout_cache.build(code)
