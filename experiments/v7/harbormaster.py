"""Add short-route recovery and choose another root plan only on an observed economic trigger."""
import inspect

import depot_corridor
import late_hire
import master_worker
import quantile
import receipt_motion
import receipt_tail
import rollout_cache
import shipment_warning


def _v7_shipping_alert(root, original, side, limit, width):
    if root['step'] < 70 or width < 4:
        return original, limit
    adjusted, raised = _v7_root_shipping(root, original, side, limit, width)
    warning = _v7_receipt_tail(root)
    if warning is None or warning[1-side, 0] < _V7_LARGE_WARNING:
        return original, limit
    return adjusted, raised


def build(code):
    code = depot_corridor.build(code)
    code = late_hire.build(code)
    code = receipt_tail.build(receipt_motion.build(code))
    motion = "        if left>0:\n            x=_v7_motion_features"
    if code.count(motion) != 1:
        raise ValueError('Missing motion-conditioned deposit estimate')
    # Public movement history is reliable at the root. Simulated child movements
    # depend on the rollout model and have no training calibration of their own.
    code = code.replace(motion, motion.replace('if left>0:', "if left>0 and '_st' not in board:"))
    code = shipment_warning.build(code, .3799)
    marker = 'def _v7_modes(obs,flow):'
    original = 'flow,limit=_v7_root_shipping(root,_v7_flows(root),me,limit,most)'
    if code.count(original) != 1 or code.count(marker) != 1:
        raise ValueError('Missing joint shipment forecast in the search root')
    code = code.replace(marker, inspect.getsource(_v7_shipping_alert) + '\n\n' + marker)
    code = code.replace(original, 'flow,limit=_v7_shipping_alert(root,_v7_flows(root),me,limit,most)')
    code = quantile.build(master_worker.build(code))
    broad = "step>=10 and (_mw_us[0]<125 or _mw_foe[0]<100 or any(u[2]=='COLLECTOR' and u[1]>=90 for u in _mw_foe[2].values()))"
    trigger = "10<=step<=185 and _mw_us[0]<60 and len(_mw_us[3])<=2 and len(_mw_foe[2])>=len(_mw_us[2])+2"
    if code.count(broad) != 1:
        raise ValueError('Missing learned worker proposal')
    return rollout_cache.build(code.replace(broad, trigger))
