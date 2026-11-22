"""Use a measured channels-last CPU convolution for small inference batches."""
import inspect


class _V7ChannelsLast(_TorchNet if '_TorchNet' in globals() else object):
    def __init__(self,original):
        self.__dict__.update(original.__dict__)
        convert=lambda w:w.contiguous(memory_format=_torch.channels_last)
        self.stem=(convert(self.stem[0]),self.stem[1])
        self.blocks=[(convert(c1w),c1b,gw,gb,convert(c2w),c2b) for c1w,c1b,gw,gb,c2w,c2b in self.blocks]

    @staticmethod
    def _conv(x,w,b):
        return _F.conv2d(_F.pad(x.contiguous(memory_format=_torch.channels_last),(1,1,1,1),mode='circular'),w,b)


class _V7BatchNet:
    def __init__(self,small,wide,threshold):
        self.small=small
        self.wide=wide
        self.threshold=threshold

    def __call__(self,x,*args,**kwargs):
        net=self.small if len(x)<=self.threshold else self.wide
        return net(x,*args,**kwargs)


def _v7_batch_net(standard,wide):
    small=_V7ChannelsLast(standard)
    rng=np.random.RandomState(359);threshold=0
    for size in (1,2,4):
        x=(rng.uniform(size=(size,NPL,15,15))<.13).astype('f')
        sid=np.zeros(size,dtype=int)
        s=np.zeros((size,5),'f') if hasattr(standard,'v1') else None
        expected=wide(x,sid,s);actual=small(x,sid,s)
        if any(not np.allclose(a,b,rtol=.001,atol=.0012) for a,b in zip(expected,actual) if a is not None):break
        elapsed=[[],[]]
        for i in (0,1,1,0,0,1):
            net=(wide,small)[i];begin=_time.perf_counter();net(x,sid,s)
            elapsed[i].append(_time.perf_counter()-begin)
        if np.median(elapsed[1])>.85*np.median(elapsed[0]):break
        threshold=size
    return _V7BatchNet(small,wide,threshold) if threshold else wide


def build(code):
    anchor='def _build(pol, val):';old='                net = w\n'
    if code.count(anchor)!=1 or code.count(old)!=1:raise ValueError('expected incumbent convolution verifier')
    body='\n\n'.join(inspect.getsource(fn) for fn in (_V7ChannelsLast,_V7BatchNet,_v7_batch_net))
    body=body.replace("_TorchNet if '_TorchNet' in globals() else object","_TorchNet")
    return code.replace(anchor,body+'\n\n'+anchor).replace(old,'                net = _v7_batch_net(net,w)\n')
