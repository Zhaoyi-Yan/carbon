"""Make near-term economic comparisons using the modal joint continuation as well as samples."""


def build(code,policy='modal'):
    if policy not in ('modal','low_temperature','alternating'):raise ValueError(policy)
    anchor='    choices = (rng.random_sample(len(cells))[:, None] >= cumulative).sum(axis=1)'
    if code.count(anchor)!=1:raise ValueError('expected row sampler')
    code=code.replace(anchor,'    choices = rows.argmax(axis=1) if rng is None else (rng.random_sample(len(cells))[:, None] >= cumulative).sum(axis=1)')
    anchor='            moves=[_sample(pl,s,w[2*j+s],b[2*j+s],np.random.RandomState(seed+373*t+17*s),t) for s in (0,1)]'
    if code.count(anchor)!=1:raise ValueError('expected the paired small-policy tail')
    scale='1.' if policy!='low_temperature' else '1.85'
    rng={'modal':'None','low_temperature':'np.random.RandomState(seed+373*t+17*s)','alternating':'None if tick%2 else np.random.RandomState(seed+373*t+17*s)'}[policy]
    return code.replace(anchor,'            moves=[_sample(pl,s,w[2*j+s]*'+scale+',b[2*j+s]*'+scale+','+rng+',t) for s in (0,1)]')
