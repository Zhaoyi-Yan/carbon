"""Lossless frozen policy plus per-output-channel 4/6-bit block deltas for a second Carbon policy."""
import argparse
import base64
import hashlib
import inspect
import io
import json
import lzma
from pathlib import Path
import re
import numpy as np

GROUP=108
def encoded(source,key='_WEIGHTS'):
    if isinstance(source,Path):source=source.read_text()
    m=re.search(r'^'+re.escape(key)+r'''\s*=\s*["']([^"'\n]+)["']\s*$''',source,re.M)
    if m is None:raise ValueError('Missing '+key)
    return m.group(1)
def arrays(encoded_npz):
    with np.load(io.BytesIO(base64.b64decode(encoded_npz)),allow_pickle=False) as f:
        return {k:f[k] for k in f.files}
def pack_codes(values,bits):
    codes=np.asarray(values,np.uint8).ravel()
    g=2 if bits==4 else 4
    if len(codes)%g:codes=np.pad(codes,(0,-len(codes)%g))
    q=codes.reshape(-1,g).T
    if bits==4:return q[0]|(q[1]<<4)
    return np.stack((q[0]|(q[1]<<6),(q[1]>>2)|(q[2]<<4),(q[2]>>4)|(q[3]<<2)),1).ravel()

def _dual_unpack(q,bits,n):
    import numpy as np
    q=np.asarray(q,np.uint8)
    if bits==4:return np.stack((q&15,q>>4),1).ravel()[:n]
    a,b,c=q.reshape(-1,3).T
    return np.stack((a&63,(a>>6)|((b&15)<<2),(b>>4)|((c&3)<<4),c>>2),1).ravel()[:n]

def _dual_harbor_weights(base_weights):
    import base64
    import hashlib
    import io
    import lzma
    import numpy as np
    original=base64.b64decode(base_weights)
    if hashlib.sha256(original).hexdigest()!=_DUAL_BASE_SHA:
        raise ValueError('Unexpected frozen base policy weights')
    delta=lzma.decompress(base64.b85decode(_DUAL_DELTA))
    output={}
    with np.load(io.BytesIO(original),allow_pickle=False) as base,np.load(io.BytesIO(delta),allow_pickle=False) as spec:
        for i,k in enumerate(base.files):
            num=str(i)
            if 'x'+num in spec:
                output[k]=spec['x'+num]
            elif 'm'+num not in spec:
                output[k]=base[k]
            else:
                w=base[k];dst=w.copy().reshape(len(w),-1);fan=dst.shape[1];mask=spec['m'+num].astype(bool)
                for bits,sel in ((4,~mask),(6,mask)):
                    rows=int(sel.sum())
                    if not rows:continue
                    tag=str(bits)+'_'+num
                    codes=_dual_unpack(spec['q'+tag],bits,rows*fan).reshape(rows,fan)
                    scale=np.repeat(spec['s'+tag].astype(np.float32),_DUAL_GROUP,axis=1)[:,:fan]
                    dst[sel]+=spec['t'+tag][codes]*scale
                output[k]=dst.reshape(w.shape)
    result=io.BytesIO()
    np.savez(result,**output)
    return base64.b64encode(result.getvalue()).decode('ascii')

def fit_blocks(delta,bits):
    n,fan=delta.shape;ng=(fan+GROUP-1)//GROUP
    padded=np.pad(delta,((0,0),(0,ng*GROUP-fan))).reshape(n,ng,GROUP)
    count=np.minimum(GROUP,np.maximum(1,fan-GROUP*np.arange(ng)))
    scale=np.sqrt(np.sum(padded.astype(float)**2,axis=2)/count).astype(np.float32)
    scale=np.maximum(scale,1.e-10)
    def flat_scale():return np.repeat(scale,GROUP,axis=1)[:,:fan]
    normalized=(delta/flat_scale()).ravel()
    # Lloyd-Max levels fit the full layer; each channel/block retains its own gain.
    levels=1<<bits
    book=np.linspace(-3.2,3.2,levels,dtype=np.float32)
    for _ in range(14):
        bins=np.searchsorted((book[1:]+book[:-1])*.5,normalized)
        counts=np.bincount(bins,minlength=levels)
        total=np.bincount(bins,weights=normalized,minlength=levels)
        book=np.where(counts,total/np.maximum(1,counts),book).astype(np.float32)
    for _ in range(2):
        codes=np.searchsorted((book[1:]+book[:-1])*.5,delta/flat_scale()).astype(np.uint8)
        ref=np.pad(book[codes],((0,0),(0,ng*GROUP-fan))).reshape(n,ng,GROUP)
        num=np.sum(padded*ref,axis=2);den=np.sum(ref*ref,axis=2)
        scale=np.where(den>0,np.maximum(1.e-10,num/np.maximum(den,1.e-20)),scale)
    # Optimize and reassign against the exact FP16 gains stored in the archive.
    scale=scale.astype(np.float16).astype(np.float32)
    codes=np.searchsorted((book[1:]+book[:-1])*.5,delta/np.maximum(flat_scale(),1.e-20)).astype(np.uint8)
    error=np.mean((delta-book[codes]*flat_scale())**2,axis=1)
    return codes,scale.astype(np.float16),book,error

def prepare(v6_source,harbor_source):
    original=encoded(v6_source);target=encoded(harbor_source)
    z=arrays(original);y=arrays(target)
    if list(z)!=list(y):raise ValueError('Policy archive keys/order differ')
    layers={};others={}
    for i,k in enumerate(z):
        v,w=z[k],y[k]
        if np.array_equal(v,w) and v.dtype==w.dtype:continue
        if v.shape!=w.shape or w.ndim<2 or w.size<4096 or w.dtype!=np.float32:
            others['x'+str(i)]=w;continue
        delta=(w-v).reshape(len(v),-1);fit={bits:fit_blocks(delta,bits) for bits in (4,6)}
        layers[i]=(delta.shape,fit)
    return hashlib.sha256(base64.b64decode(original)).hexdigest(),layers,others

def render(prepared,fraction):
    checksum,layers,others=prepared;spec=dict(others);ranking=[]
    for key,((n,fan),fit) in layers.items():
        # Rank accuracy gain per extra coded scalar, independent of validation data.
        for row,benefit in enumerate(fit[4][3]-fit[6][3]):ranking.append((float(benefit),key,row,fan))
    budget=fraction*sum(n*fan for (n,fan),_ in layers.values())
    masks={key:np.zeros(shape[0],np.uint8) for key,(shape,_) in layers.items()};used=0
    for gain,key,row,cost in sorted(ranking,reverse=True):
        if used+cost>budget:continue
        masks[key][row]=1;used+=cost
    for key,((n,fan),fit) in layers.items():
        mask=masks[key];spec['m'+str(key)]=mask
        for bits,take in ((4,mask==0),(6,mask==1)):
            if not take.any():continue
            tag=str(bits)+'_'+str(key);codes,scale,table,_=fit[bits]
            spec['q'+tag]=pack_codes(codes[take],bits);spec['s'+tag]=scale[take];spec['t'+tag]=table
    buf=io.BytesIO();np.savez(buf,**spec)
    payload=base64.b85encode(lzma.compress(buf.getvalue(),preset=4)).decode('ascii')
    code=(f'_DUAL_BASE_SHA={checksum!r}\n_DUAL_GROUP={GROUP}\n_DUAL_DELTA={payload!r}\n'
          +inspect.getsource(_dual_unpack)+'\n'+inspect.getsource(_dual_harbor_weights)+'\n')
    return code,{'characters':len(code),'six_bit_scalars':used,'quantized_scalars':sum(n*fan for (n,fan),_ in layers.values()),'requested_six_fraction':fraction}

def _dual_frozen_base85(chars,length):
    import base64
    import numpy as np
    alphabet=b'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~'
    lookup=np.zeros(256,np.uint32)
    lookup[np.frombuffer(alphabet,np.uint8)]=np.arange(85,dtype=np.uint32)
    digits=np.frombuffer(chars.encode('ascii'),np.uint8).reshape(-1,5)
    val=lookup[digits[:,0]]
    for j in range(1,5):
        val*=np.uint32(85)
        val+=lookup[digits[:,j]]
    raw=val.astype('>u4',copy=False).tobytes()[:length]
    return base64.b64encode(raw).decode('ascii')

def build_frozen_snippet(v6_source):
    """Define _WEIGHTS with precisely the original NPZ bytes, stored as Base85."""
    raw=base64.b64decode(encoded(v6_source))
    payload=base64.b85encode(raw,pad=True).decode('ascii')
    return inspect.getsource(_dual_frozen_base85)+f'\n_WEIGHTS = _dual_frozen_base85({payload!r},{len(raw)})\n'

def build_pair(v6_source,harbor_source,total_chars=5912710,delta_chars=700000):
    """Replace the original _WEIGHTS assignment with this single, self-contained fragment."""
    frozen=build_frozen_snippet(v6_source)
    return frozen+build_snippet(v6_source,harbor_source,min(delta_chars,total_chars-len(frozen)))

def build_snippet(v6_source,harbor_source,max_chars=700000):
    """Return only the self-contained initializer for _dual_harbor_weights(_WEIGHTS)."""
    p=prepare(v6_source,harbor_source)
    winner=None
    for f in (1.,.75,.5,.35,.25,.15,0.):
        code,_=render(p,f)
        if len(code)<=max_chars:winner=code;break
    if winner is None:raise ValueError('Source-character budget is smaller than the all-4-bit representation')
    return winner

def main(args):
    frozen=build_frozen_snippet(args.base) if args.pair else ''
    allowance=min(args.limit,args.total_limit-len(frozen)) if args.pair else args.limit
    p=prepare(args.base,args.target);stats=[];best=None
    for f in (1.,.75,.5,.35,.25,.15,0.):
        code,s=render(p,f);stats.append(s)
        if best is None and len(code)<=allowance:best=code
    if best is None:raise ValueError('Requested source limit is too small')
    args.out.write_text(frozen+best)
    print(json.dumps(dict(frozen_chars=len(frozen),delta_chars=len(best),total_chars=len(frozen+best),candidates=stats)),flush=True)
if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('base','target','out'):cli.add_argument(k,type=Path)
    cli.add_argument('--limit',type=int,default=700000,help='Maximum characters for the differential initializer')
    cli.add_argument('--pair',action='store_true',help='Emit the bit-exact frozen _WEIGHTS initializer as well')
    cli.add_argument('--total-limit',type=int,default=5912710);main(cli.parse_args())
