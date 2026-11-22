"""Bayesian carbon estimates from legal vision, public harvests and tree income."""
_V5_PARTICLES = np.arange(44,dtype=float)
_V5_CLASSES = np.asarray(CLS)
_V5_CELL = np.arange(225)
_V5_NEAR = np.asarray(NB)
_V5_MOORE = np.array([[p]+NB8[p] for p in range(225)])


class _V5Map:
    def __init__(self, players):
        # Official 15x15 generation chooses 29 of 64 quadrant cells and draws stock
        # uniformly between 1 and 2*(2500/4/29) before rounding down to integers.
        maximum = 1250./29.
        occupied = np.maximum(0.,np.minimum(_V5_PARTICLES+1,maximum)-np.maximum(_V5_PARTICLES,1.))
        occupied /= occupied.sum()
        prior = occupied * (29./64.)
        prior[0] += 35./64.
        self.weights = np.tile(prior,(64,1))
        self.values = np.tile(_V5_PARTICLES,(225,1))
        self.home = [p for side in players for p in side[1].values()]
        self.values[self.home] = 0
        self.last = None
        self.sticky = np.zeros(225,dtype=bool)
        self.moves = {}

    def _learn(self, pos, particles, observed, width):
        if self.sticky[pos] or np.ptp(particles)<.025:
            return
        differences = abs(particles-observed)
        if differences.min()>max(.36,2.5*width):
            return
        w = self.weights[_V5_CLASSES[pos]]
        likely = np.exp(-.5*(differences/width)**2)
        # Keep small support for ambiguous movement and rounded balances.
        updated = w*(.0001+likely)
        if updated.sum() > 1e-8:
            w[:] = updated/updated.sum()

    def observe(self, obs, side):
        players, step = obs['players'], obs['step']
        tree = {p:(who,tid,age) for who,pl in enumerate(players) for tid,(p,age) in pl[3].items()}
        units = {uid:(p,v,kind,who) for who,pl in enumerate(players) for uid,(p,v,kind) in pl[2].items()}
        if self.last is not None:
            previous, oldtrees = self.last
            v = self.values
            died = {uid:u for uid,u in previous.items() if uid not in units}
            uncertain = set()
            for uid,(p,cargo,kind,owner) in died.items():
                if owner==side:
                    uncertain.add(move(p,WIDX.get(self.moves.get(uid),0)))
                else:
                    uncertain.update(N1[p])
            at = {u[0]:uid for uid,u in units.items()}
            # Only a collision without a surviving collector can spill onto the ground.
            for p in uncertain:
                if p not in at and p not in tree and p not in self.home:
                    self.sticky[p] = True
            stay = {p for uid,(p,_,_,_) in units.items() if uid not in previous or previous[uid][0]==p}
            stationary = {p:uid for p,uid in at.items() if p in stay and units[uid][2]=='COLLECTOR'}
            oldv = v.copy()
            blocked = np.zeros(225,bool)
            absorption = np.zeros(225,int)
            tax = np.zeros((2,225),int)
            for p,(who,tid,age) in tree.items():
                if age>1:
                    tax[1-who,_V5_MOORE[p]]+=1
            for p,uid in stationary.items():
                q = units[uid]
                if p in tree or p in self.home:
                    continue
                if uid in previous and p not in uncertain and tax[q[3],p]<10:
                    harvested = q[1]/(1.-.1*tax[q[3],p])-previous[uid][1]
                    self._learn(p,oldv[p],harvested/.3,.12)
                v[p] *= .7
            for p in oldtrees.keys()-tree.keys():
                v[p]=30.
                self.sticky[p]=False
                blocked[p]=True
            means=(oldv*self.weights[_V5_CLASSES]).sum(1)
            variance=((oldv-means[:,None])**2*self.weights[_V5_CLASSES]).sum(1)
            for p,(who,tid,age) in tree.items():
                blocked[_V5_NEAR[p]]=True
                v[p]=0.
                blocked[p]=True
                self.sticky[p]=False
                if age<=1:
                    continue
                cells=[q for q in NB[p] if q not in tree and q not in stationary and q not in self.home]
                absorption[cells]+=1
                receipt=obs.get('trees',{}).get(tid)
                if not receipt or not cells or any(q in uncertain or self.sticky[q] for q in cells):
                    continue
                taxable=[units[at[q]] for q in _V5_MOORE[p] if q in at and units[at[q]][3]!=who and units[at[q]][2]=='COLLECTOR']
                if any(tax[u[3],u[0]]>=10 for u in taxable):
                    continue
                paid=sum(.1*u[1]/(1.-.1*tax[u[3],u[0]]) for u in taxable)
                total=max(0.,(receipt[1]-paid)/.0375)
                group={}
                for q in cells: group.setdefault(_V5_CLASSES[q],[]).append(q)
                for same in group.values():
                    rest=[q for q in cells if q not in same]
                    width=(.36+sum(variance[q] for q in rest))**.5
                    self._learn(same[0],oldv[same].sum(0),total-sum(means[q] for q in rest),width)
            blocked[list(stay)]=True
            # Unknown opponent death cells can cover several possibilities; only
            # stop growth on a confirmed position without a surviving worker.
            v *= (1.-.0375*absorption[:,None])
            v[~blocked]=np.minimum(100.,v[~blocked]*1.03)
            np.round(v,3,out=v)
            v[self.home]=0.
        visible=set()
        for p,_,role in players[side][2].values():
            if role=='COLLECTOR':visible.add(p)
            else:visible.update(NB[p])
        for p in visible:
            carbon=float(obs['carbon'][p])
            self._learn(p,self.values[p],carbon,.075)
            self.values[p]=carbon
            self.sticky[p]=False
        self.last=units,tree
        est=(self.values*self.weights[_V5_CLASSES]).sum(1)
        return est

    def sample(self,rng):
        # Draw once per symmetry class. All proposals within this rollout use the same board.
        u=rng.random_sample(64)
        indices=(self.weights.cumsum(1)<u[:,None]).sum(1).clip(0,len(_V5_PARTICLES)-1)
        return self.values[_V5_CELL,indices[_V5_CLASSES]].tolist()
