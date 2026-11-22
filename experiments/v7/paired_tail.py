"""Build the common-reply tail evaluation for frozen Lifeline."""
import inspect

FROZEN = '9c5bdda7b8d1a9cc33af4cd0fe03edcb4a34a2d42e027d9b06fed519ff6b949d'


def _v7_tail_over(board):
    return board['step'] > LAST_TURN or board.get('_terminal_sign') is not None


def _v7_tail_pairs(scored, candidates, replies, expected_step):
    cells={}
    for c,m,board,previous in scored:
        if c in candidates and m in replies:
            cells.setdefault(m,{})[c]=[c,m,board,previous]
    groups=[];seen=set()
    for m in sorted(cells):
        rows=cells[m]
        if any(c not in rows for c in candidates):continue
        pair=[rows[c] for c in candidates]
        if any(b[2]['step']!=expected_step and not (_v7_tail_over(b[2]) and b[2]['step']<=expected_step) for b in pair):continue
        key=tuple(sorted((u,a) for u,a in replies[m].items() if a))
        if key in seen:continue
        seen.add(key);groups.append(pair)
    return groups


def _v7_tail_extend(groups,me,config,t0,seed,allowance):
    # The original tail spends at most 14 candidate-pair ticks. Advance all
    # distinct enemy replies in lockstep, keeping that same total work cap.
    rows=[state for pair in groups for state in pair];n=len(groups)
    completed=0;per_tick=.016*n
    try:
        for _ in range(14//n):
            live=[s for s in rows if not _v7_tail_over(s[2])]
            if not live:break
            now=_time.perf_counter()
            if now-t0+max(.022,1.45*per_tick)>allowance:break
            views=[]
            for _,_,board,previous in live:
                if 'v5_views' not in board:
                    board['v5_views']=_v5_pair(board['players'],board['step'],board['carbon'],previous)
                views.extend(board['v5_views'])
            w,b,_=_V6R_TINY(np.asarray(views),_v7_rollout_ids([0,1]*len(live)))
            pending=[]
            for j,(_,_,board,_) in enumerate(live):
                if _time.perf_counter()-t0>=allowance:break
                pl=board['players'];t=board['step']
                act=[_sample(pl,s,w[2*j+s],b[2*j+s],np.random.RandomState(seed+373*t+17*s),t) for s in (0,1)]
                pending.append((pl,_turn(board,config,act)))
            if len(pending)!=len(live):break
            for st,(pl,out) in zip(live,pending):st[3],st[2]=pl,out
            completed+=1;per_tick=.55*per_tick+.45*(_time.perf_counter()-now)
        if completed<3 and any(not _v7_tail_over(s[2]) for s in rows):return None
        # Reserve time for all four leaf values. Never blend an incomplete pair.
        if _time.perf_counter()-t0+.004*n>allowance:return None
        totals=[0.,0.]
        for pair in groups:
            for j,(_,_,board,_) in enumerate(pair):
                if _time.perf_counter()-t0>=allowance:return None
                sign=board.get('_terminal_sign')
                if sign is not None:v=3.*sign*(1-2*me)
                elif board['step']>LAST_TURN:
                    cash=[p[0] for p in board['players']];v=_terminal(cash[me]-cash[1-me])
                else:v=_r6_asset_score(board,me)
                if not np.isfinite(v):return None
                totals[j]+=v
        return [v/n for v in totals]
    except Exception:
        # Discard a failed joint tail; the already-complete main-search means survive.
        return None


def _v7_paired_tail(legacy,scored,candidates,me,config,t0,seed,allowance,replies,expected_step):
    if len(replies)<=1:return _r6_tail(legacy,candidates,me,config,t0,seed,allowance)
    groups=_v7_tail_pairs(scored,candidates,replies,expected_step)
    if len(groups)<2:
        if not groups:return None
        if groups[0][0][1]==0:return _r6_tail(legacy,candidates,me,config,t0,seed,allowance)
        single=[[c,0,board,prev] for c,_,board,prev in groups[0]]
        return _r6_tail(single,candidates,me,config,t0,seed,allowance)
    return _v7_tail_extend(groups,me,config,t0,seed,allowance)


def build(code):
    """Keep the original one-reply routine intact and preserve only fully scored snapshots."""
    def replace(text,before,after):
        if text.count(before)!=1:raise ValueError('Unexpected Lifeline search source: '+before)
        return text.replace(before,after)
    routines='\n\n'.join(inspect.getsource(fn).strip() for fn in (_v7_tail_over,_v7_tail_pairs,_v7_tail_extend,_v7_paired_tail))
    code=replace(code,'def _r6_tail(boards, candidates, me, config, t0, seed, allowance):',routines+'\n\n\ndef _r6_tail(boards, candidates, me, config, t0, seed, allowance):')
    code=replace(code,'    boards = begin(alive, range(M))','    scored_tail = []\n    boards = begin(alive, range(M))')
    code=replace(code,'        depth = t\n        if t == D2','        depth = t\n        scored_tail = [b[:] for b in run]\n        if t == D2')
    code=replace(code,'        rolls += M\n    dt =','        scored_tail.extend(b[:] for b in new)\n        rolls += M\n    dt =')
    return replace(code,'tail=_r6_tail(boards,two,me,configuration,t0,seed0,.83)',
                        'tail=_v7_paired_tail(boards,scored_tail,two,me,configuration,t0,seed0,.83,{m:opp[m] for m in range(rolls)},step+depth)')
