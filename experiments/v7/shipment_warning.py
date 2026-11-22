"""Use the large-receipt model to generate both sides of an urgent depot race at the search root."""
import inspect
import numpy as np

def _v7_root_shipping(root,flow,side,limit,width):
    tail=_v7_receipt_tail(root)
    if tail is None:return flow,limit
    result=flow.copy()
    # The per-team calibration measures all future deposits, including newly mined cargo.
    result[:,1]=np.maximum(result[:,0],.30*result[:,1]+.70*tail[:,2])
    if root['step']>=70 and width>=4:
        other=1-side
        # Keep both the return and interception alternatives available in ordinary cash races.
        threshold=_V7_LARGE_WARNING
        if tail[other,0]>=threshold and result[side,1]>=70.:
            limit=max(2,limit)
    return result,limit


def build(code,threshold):
    before='        alternatives=_v7_proposals(obs,configuration,est,act0,fixed or {},_v7_flows(root),limit)'
    marker='def _v7_modes(obs,flow):'
    if code.count(before)!=1 or code.count(marker)!=1:raise ValueError('unexpected root transport chooser')
    return code.replace(marker,'_V7_LARGE_WARNING='+repr(float(threshold))+'\n'+inspect.getsource(_v7_root_shipping)+'\n\n'+marker).replace(before,"        flow,limit=_v7_root_shipping(root,_v7_flows(root),me,limit,most)\n"+before.replace('_v7_flows(root)','flow'))
