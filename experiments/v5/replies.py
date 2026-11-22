"""Compare candidate moves against common opponent replies before expensive trajectories."""
def _v5_reply_adjustments(root,config,candidates,seed,t0):
    me=root['player'];other=1-me
    workers,bases=_ST['v5_history']['opponent']
    at=root['step'];players=root['players']
    replies=[];values=[]
    for i,temp in enumerate((1.,.2,1.65,2.5)):
        rng=np.random.RandomState(seed+7+(311*i))
        reply=_sample(players,other,workers/temp,bases/temp,rng,at)
        if reply in replies:continue
        if i and _time.perf_counter()-t0>.18:break
        scores=[]
        for action in candidates:
            if _time.perf_counter()-t0>.25:break
            moves=[action,reply] if me==0 else [reply,action]
            board=_turn(root,config,moves)
            # The 48-turn trained projection is cheap enough for all candidate/reply pairs.
            pred=_v5_cash_only(board,me)
            scores.append(pred)
        if len(scores)!=len(candidates):break
        scores=np.asarray(scores)
        values.append(scores-scores[0]);replies.append(reply)
    if len(values)<2:return np.zeros(len(candidates))
    scores=np.stack(values)
    late=max(0.,1.-(LAST_TURN-at)/100.)
    lead=np.tanh((players[me][0]-players[other][0])/900.)
    risk=.22+.20*max(0.,lead)*late
    expectation=(1.-risk)*scores.mean(0)+risk*scores.min(0)
    # Correct the existing simulation's first reply while retaining a small
    # contribution from expected material even if the replies agree.
    return .62*(expectation-.70*scores[0])
