"""Keep Soundings' search in ordinary positions and add two event-triggered responses."""
import inspect
import depot_corridor
import receipt_tail
import shipment_warning


def _v7_late_alarm(root, flow, side, limit, width):
    if root['step']<150 or width<4 or flow[1-side,1]<125 or flow[side,1]<65:
        return flow,limit
    warning=_v7_receipt_tail(root)
    if warning is None or warning[1-side,0]<_V7_LARGE_WARNING:
        return flow,limit
    return _v7_root_shipping(root,flow,side,limit,width)


def build(code):
    code=depot_corridor.build(code)
    code=shipment_warning.build(receipt_tail.build(code),.3799)
    hook='flow,limit=_v7_root_shipping(root,_v7_flows(root),me,limit,most)'
    anchor='def _v7_modes(obs,flow):'
    if code.count(hook)!=1 or code.count(anchor)!=1:raise ValueError('Unknown Soundings search')
    return code.replace(anchor,inspect.getsource(_v7_late_alarm)+'\n\n'+anchor).replace(
        hook,'flow,limit=_v7_late_alarm(root,_v7_flows(root),me,limit,most)')
