"""An optional more dispersed Apollo control; common random numbers across branch arms."""
import numpy as np
if __package__:
    from .surrogate import Surrogate, make as make_primary
else:
    from surrogate import Surrogate, make as make_primary

TEMPERATURE=.55


def perturb(logits,bases,seed,step,side):
    # The draw is a function of the map seed, turn, seat, tile and action.
    # Different branches at the same turn use common randomness, even after a unit dies.
    key=(int(seed)*1000003 + int(step)*104729 + int(side)*130363 + 6101) % (2**32-1)
    rng=np.random.RandomState(key)
    return (logits+TEMPERATURE*rng.gumbel(size=logits.shape),
            bases+TEMPERATURE*rng.gumbel(size=bases.shape))


class VariedApollo(Surrogate):
    def __init__(self,seed,config):
        super().__init__('apollo',seed,config)

    def logits(self,obs,estimate,known):
        worker,base=super().logits(obs,estimate,known)
        return perturb(worker,base,self.seed,obs['step'],obs['player'])


def make(name,seed,config):
    if name=='apollo-varied':return VariedApollo(seed,config)
    return make_primary(name,seed,config)
