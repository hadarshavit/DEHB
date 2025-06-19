from ConfigSpace import Configuration, ConfigurationSpace, Float
from src.dehb.optimizers.de import DE
import numpy as np


class QuadraticFunction:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x = Float("x", (-5, 5), default=-5)
        cs.add([x])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        x = config["x"]
        res = {
            "fitness": x**2,  # must-have key that DE/DEHB minimizes
            "cost": 0,  # must-have key that associates cost/runtime
            "info": dict(),  # optional key containing a dictionary of additional info
        }
        # print(x **2)
        return res


if __name__ == "__main__":
    bench = QuadraticFunction()
    cs = ConfigurationSpace(seed=0)
    x = Float("x", (-5, 5), default=-5)
    cs.add([x])

    de = DE(cs=cs, f=bench.train, seed=0, pop_size=10, mutation_factor=0.5, crossover_prob=0.5)
    orig = de.run(generations=9, verbose=False)[0]
    # de.run(generations=10, verbose=False)
    cs = ConfigurationSpace(seed=0)
    x = Float("x", (-5, 5), default=-5)
    cs.add([x])
    print("*"   *50)
    de = DE(cs=cs, seed=0, pop_size=10, mutation_factor=0.5, crossover_prob=0.5)

    for i in range(10 * 10):
        trial, trial_id, target_idx = de.ask()
        res = bench.train(trial, seed=0)
        de.tell(trial, target_idx, res)
    mod = np.array(de.traj)

    assert np.all(orig == mod), f"Original {orig} and modified {mod} trajectories do not match."
    print("Original and modified trajectories match.")
    print("Original trajectory:", orig)

