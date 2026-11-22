"""Package a reserve policy and a pursuit policy with shared observation and income history."""
import base64
import hashlib
import inspect
import re
import textwrap

BLOBS = re.compile(r"""(['"])([A-Za-z0-9+/=]{1000,})\1""")


def _v7_choose_team(obs, state):
    turn, side = obs['step'], obs['player']
    if turn < 125:
        return 'pursuit'
    us, them = obs['players'][side], obs['players'][1-side]
    reserve = us[0]-them[0]
    receipts = _v7_receipts(obs['players'],turn,LAST_TURN)
    delta = receipts[side] - receipts[1-side]
    # Ten-step and five-step predictions overlap. The weights sum to one.
    projected = reserve + .45*float(delta[0]) + .55*float(delta[1])
    deprived = us[0] < 60 and len(us[3]) <= 2 and len(us[2])+2 <= len(them[2])
    convoy = receipts[1-side,0] >= 155 and receipts[1-side,1] >= receipts[side,1]+125
    if deprived or state.get('corridor_track') or (convoy and projected < 650):
        return 'pursuit'
    last = state.get('dual_team','pursuit')
    if turn-state.get('dual_switch',-30) < 12:
        return last
    if projected > 90:
        return 'reserve'
    if projected < -95:
        return 'pursuit'
    return last


def _dual_agent(observation, configuration):
    turn=observation['step']
    if not _SEARCH_OK or configuration.get('size',15)!=SIZE:
        return _dual_pursuit(observation,configuration)
    if turn==0:
        _ST.pop('dual_team',None)
        _ST.pop('dual_switch',None)
    team=_v7_choose_team(observation,_ST)
    if _ST.get('dual_team')!=team:
        _ST['dual_team'],_ST['dual_switch']=team,turn
    if team=='pursuit':
        return _dual_pursuit(observation,configuration)
    previous=_ST.get('prev')
    _v7_watch_cash(observation,previous,_ST.get('liquidity_action',{}),_ST)
    action=_dual_reserve(observation,configuration,_ST,_FC_HISTORY,_S_STATS)
    _ST['receipt_previous2']=previous
    _ST['liquidity_action']=dict(action)
    return action


def materialize(incumbent, challenger, weight_snippet, source_weights):
    # Both files contain the same public shadow network, old economic predictor and some heads.
    # Decode common payloads once. The two independently trained value heads and small
    # rollout policies remain distinct.
    raw={}
    def deduplicate(code):
        def substitute(match):
            blob=match[2]
            if blob==source_weights[0]:return '_dual_frozen_blob'
            if blob==source_weights[1]:return '_dual_refit_blob'
            key='_dual_data_'+hashlib.sha256(blob.encode()).hexdigest()[:14]
            raw[key]=blob
            return key
        return BLOBS.sub(substitute,code)
    active=deduplicate(challenger)
    reserve=deduplicate(incumbent)
    # Nested functions bind their own policy weights and rollout identifiers. Its trackers,
    # recent receipts and search counters are the ones maintained by whichever side played.
    reserve=reserve.replace('global LAST_TURN','nonlocal LAST_TURN').replace('global _NET, _RNET','nonlocal _NET, _RNET')
    reserve=reserve.replace("'next_state' in globals()","'next_state' in locals()")
    reserve=reserve.replace("globals().get('_ROLL')","False")
    shadow_start=reserve.index('    _R6_OLD_POLICY=_Policy(_SHADOW_DATA)')
    shadow_end=reserve.index('if _SEARCH_OK:',shadow_start)
    reserve=reserve[:shadow_start]+'    _R6_OLD_POLICY,_R6_OLD_NET=shared_shadow\n'+reserve[shadow_end:]
    dispatch="""
def _reserve_turn(observation, configuration, state, history, statistics):
    nonlocal _ST, _FC_HISTORY, _S_STATS
    _ST, _FC_HISTORY, _S_STATS = state, history, statistics
    return agent(observation, configuration)
return _reserve_turn
"""
    preamble=active[:active.index('import base64')]
    active=active[len(preamble):]
    payload="import base64\n"+weight_snippet+"\n"
    for key,val in raw.items():
        packed=base64.b85encode(base64.b64decode(val)).decode()
        payload+=key+'=base64.b64encode(base64.b85decode('+repr(packed)+'))\n'
    nested='\n\ndef _v7_make_reserve(shared_shadow):\n'+textwrap.indent(reserve+'\n'+dispatch,'    ')
    end="\n_dual_pursuit=agent\n_dual_reserve=_v7_make_reserve((_R6_OLD_POLICY,_R6_OLD_NET)) if _SEARCH_OK else None\n"
    out=preamble+payload+active+nested+end+inspect.getsource(_v7_choose_team)+'\n'+inspect.getsource(_dual_agent)+'\nagent=_dual_agent\n'
    compile(out,'<two-policies>','exec')
    return out


def sources(incumbent, challenger):
    def original(code):
        return re.search(r'^_WEIGHTS = "(.*?)"$',code,flags=re.M)[1]
    return original(incumbent),original(challenger)


def build(incumbent,challenger):
    from dual_weights import build_snippet
    weights=sources(incumbent,challenger)
    packed=base64.b85encode(base64.b64decode(weights[0])).decode()
    delta=build_snippet(incumbent,challenger)
    primary='_dual_frozen_blob=base64.b64encode(base64.b85decode('+repr(packed)+'))\n'
    restore='_dual_refit_blob=_dual_harbor_weights(_dual_frozen_blob)\n'
    return materialize(incumbent,challenger,primary+delta+'\n'+restore,weights)
