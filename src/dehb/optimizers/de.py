import os
from pathlib import Path
from typing import List

import ConfigSpace
import ConfigSpace.util
import numpy as np
from distributed import Client

from ..utils import ConfigRepository


class DEBase():
    '''Base class for Differential Evolution
    '''
    def __init__(self, cs=None, f=None, dimensions=None, pop_size=None, max_age=None,
                 mutation_factor=None, crossover_prob=None, strategy=None,
                 boundary_fix_type='random', config_repository=None, seed=None, **kwargs):
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        elif isinstance(seed, np.random.Generator):
            seed = int(seed.integers(0, 2**32 - 1))

        assert isinstance(seed, int)

        self._original_seed = seed
        self.rng = np.random.default_rng(self._original_seed)

        # Benchmark related variables
        self.cs = cs
        self.f = f
        if dimensions is None and self.cs is not None:
            self.dimensions = len(self.cs.get_hyperparameters())
        else:
            self.dimensions = dimensions

        # DE related variables
        self.pop_size = pop_size
        self.max_age = max_age
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
        self.fix_type = boundary_fix_type

        # Miscellaneous
        self.configspace = True if isinstance(self.cs, ConfigSpace.ConfigurationSpace) else False
        self.hps = dict()
        if self.configspace:
            self.cs.seed(self._original_seed)
            for i, hp in enumerate(cs.get_hyperparameters()):
                # maps hyperparameter name to positional index in vector form
                self.hps[hp.name] = i
        self.output_path = Path(kwargs["output_path"]) if "output_path" in kwargs else Path("./")
        self.output_path.mkdir(parents=True, exist_ok=True)

        if config_repository:
            self.config_repository = config_repository
        else:
            self.config_repository = ConfigRepository()

        # Global trackers
        self.inc_score : float
        self.inc_config : np.ndarray[float]
        self.inc_id : int
        self.population : np.ndarray[np.ndarray[float]]
        self.population_ids :np.ndarray[int]
        self.fitness : np.ndarray[float]
        self.age : int
        self.history : list[object]
        self.reset()

    def reset(self, *, reset_seeds: bool = True):
        self.inc_score = np.inf
        self.inc_config = None
        self.inc_id = -1
        self.population = None
        self.population_ids = None
        self.fitness = None
        self.age = None

        if reset_seeds:
            if isinstance(self.cs, ConfigSpace.ConfigurationSpace):
                self.cs.seed(self._original_seed)
            self.rng = np.random.default_rng(self._original_seed)

        self.history = []

    def _shuffle_pop(self):
        pop_order = np.arange(len(self.population))
        self.rng.shuffle(pop_order)
        self.population = self.population[pop_order]
        self.fitness = self.fitness[pop_order]
        self.age = self.age[pop_order]

    def _sort_pop(self):
        pop_order = np.argsort(self.fitness)
        self.rng.shuffle(pop_order)
        self.population = self.population[pop_order]
        self.fitness = self.fitness[pop_order]
        self.age = self.age[pop_order]

    def _set_min_pop_size(self):
        if self.mutation_strategy in ['rand1', 'rand2dir', 'randtobest1']:
            self._min_pop_size = 3
        elif self.mutation_strategy in ['currenttobest1', 'best1']:
            self._min_pop_size = 2
        elif self.mutation_strategy in ['best2']:
            self._min_pop_size = 4
        elif self.mutation_strategy in ['rand2']:
            self._min_pop_size = 5
        else:
            self._min_pop_size = 1

        return self._min_pop_size

    def init_population(self, pop_size: int) -> List:
        if self.configspace:
            # sample from ConfigSpace s.t. conditional constraints (if any) are maintained
            population = self.cs.sample_configuration(size=pop_size)
            if not isinstance(population, List):
                population = [population]
            # the population is maintained in a list-of-vector form where each ConfigSpace
            # configuration is scaled to a unit hypercube, i.e., all dimensions scaled to [0,1]
            population = [self.configspace_to_vector(individual) for individual in population]
        else:
            # if no ConfigSpace representation available, uniformly sample from [0, 1]
            population = self.rng.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))

        return np.array(population)

    def sample_population(self, size: int = 3, alt_pop: List = None) -> List:
        '''Samples 'size' individuals

        If alt_pop is None or a list/array of None, sample from own population
        Else sample from the specified alternate population (alt_pop)
        '''
        if isinstance(alt_pop, list) or isinstance(alt_pop, np.ndarray):
            idx = [indv is None for indv in alt_pop]
            if any(idx):
                selection = self.rng.choice(np.arange(len(self.population)), size, replace=False)
                return self.population[selection]
            else:
                if len(alt_pop) < 3:
                    alt_pop = np.vstack((alt_pop, self.population))
                selection = self.rng.choice(np.arange(len(alt_pop)), size, replace=False)
                alt_pop = np.stack(alt_pop)
                return alt_pop[selection]
        else:
            selection = self.rng.choice(np.arange(len(self.population)), size, replace=False)
            return self.population[selection]

    def boundary_check(self, vector: np.ndarray) -> np.ndarray:
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        if fix_type == 'random', the values are replaced with a random sampling from (0,1)
        if fix_type == 'clip', the values are clipped to the closest limit from {0, 1}

        Parameters
        ----------
        vector : array

        Returns
        -------
        array
        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        if self.fix_type == 'random':
            vector[violations] = self.rng.uniform(low=0.0, high=1.0, size=len(violations))
        else:
            vector[violations] = np.clip(vector[violations], a_min=0, a_max=1)
        return vector

    def vector_to_configspace(self, vector: np.ndarray) -> ConfigSpace.Configuration:
        '''Converts numpy array to ConfigSpace object

        Works when self.cs is a ConfigSpace object and the input vector is in the domain [0, 1].
        '''
        # creates a ConfigSpace object dict with all hyperparameters present, the inactive too
        new_config = ConfigSpace.util.impute_inactive_values(
            self.cs.get_default_configuration()
        ).get_dictionary()
        # iterates over all hyperparameters and normalizes each based on its type
        for i, hyper in enumerate(self.cs.get_hyperparameters()):
            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1/len(hyper.sequence))
                param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1/len(hyper.choices))
                param_value = hyper.choices[np.where((vector[i] < ranges) == False)[0][-1]]
            elif type(hyper) == ConfigSpace.Constant:
                param_value = hyper.default_value
            else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
                # rescaling continuous values
                if hyper.log:
                    log_range = np.log(hyper.upper) - np.log(hyper.lower)
                    param_value = np.exp(np.log(hyper.lower) + vector[i] * log_range)
                else:
                    param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
                if type(hyper) == ConfigSpace.UniformIntegerHyperparameter:
                    param_value = int(np.round(param_value))  # converting to discrete (int)
                else:
                    param_value = float(param_value)
            new_config[hyper.name] = param_value
        # the mapping from unit hypercube to the actual config space may lead to illegal
        # configurations based on conditions defined, which need to be deactivated/removed
        new_config = ConfigSpace.util.deactivate_inactive_hyperparameters(
            configuration = new_config, configuration_space=self.cs
        )
        return new_config

    def configspace_to_vector(self, config: ConfigSpace.Configuration) -> np.ndarray:
        '''Converts ConfigSpace object to numpy array scaled to [0,1]

        Works when self.cs is a ConfigSpace object and the input config is a ConfigSpace object.
        Handles conditional spaces implicitly by replacing illegal parameters with default values
        to maintain the dimensionality of the vector.
        '''
        # the imputation replaces illegal parameter values with their default
        config = ConfigSpace.util.impute_inactive_values(config)
        dimensions = len(self.cs.get_hyperparameters())
        vector = [np.nan for i in range(dimensions)]
        for name in config:
            i = self.hps[name]
            hyper = self.cs.get_hyperparameter(name)
            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                nlevels = len(hyper.sequence)
                vector[i] = hyper.sequence.index(config[name]) / nlevels
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                nlevels = len(hyper.choices)
                vector[i] = hyper.choices.index(config[name]) / nlevels
            elif type(hyper) == ConfigSpace.Constant:
                vector[i] = 0 # set constant to 0, so that it wont be affected by mutation
            else:
                bounds = (hyper.lower, hyper.upper)
                param_value = config[name]
                if hyper.log:
                    vector[i] = np.log(param_value / bounds[0]) / np.log(bounds[1] / bounds[0])
                else:
                    vector[i] = (config[name] - bounds[0]) / (bounds[1] - bounds[0])
        return np.array(vector)

    def f_objective(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def mutation(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def crossover(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def evolve(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def run(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")


class DE(DEBase):
    def __init__(self, cs=None, f=None, dimensions=None, pop_size=20, max_age=np.inf,
                 mutation_factor=None, crossover_prob=None, strategy='rand1_bin', encoding=False,
                 dim_map=None, seed=None, config_repository=None, **kwargs):
        super().__init__(cs=cs, f=f, dimensions=dimensions, pop_size=pop_size, max_age=max_age,
                         mutation_factor=mutation_factor, crossover_prob=crossover_prob,
                         strategy=strategy, seed=seed, config_repository=config_repository,
                         **kwargs)
        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split('_')[0]
            self.crossover_strategy = self.strategy.split('_')[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None
        self.encoding = encoding
        self.dim_map = dim_map
        self._set_min_pop_size()
        # For ask-tell interface
        self._ask_queue = []  # Stores (target_idx, trial, trial_id)

    def __getstate__(self):
        """ Allows the object to picklable while having Dask client as a class attribute.
        """
        d = dict(self.__dict__)
        d["client"] = None  # hack to allow Dask client to be a class attribute
        d["logger"] = None  # hack to allow logger object to be a class attribute
        return d

    def __del__(self):
        """ Ensures a clean kill of the Dask client and frees up a port.
        """
        if hasattr(self, "client") and isinstance(self.client, Client):
            self.client.close()

    def reset(self, *, reset_seeds: bool = True):
        super().reset(reset_seeds=reset_seeds)
        self.traj = []
        self.runtime = []
        self.history = []

    def _set_min_pop_size(self):
        if self.mutation_strategy in ['rand1', 'rand2dir', 'randtobest1']:
            self._min_pop_size = 3
        elif self.mutation_strategy in ['currenttobest1', 'best1']:
            self._min_pop_size = 2
        elif self.mutation_strategy in ['best2']:
            self._min_pop_size = 4
        elif self.mutation_strategy in ['rand2']:
            self._min_pop_size = 5
        else:
            self._min_pop_size = 1

        return self._min_pop_size

    def map_to_original(self, vector):
        dimensions = len(self.dim_map.keys())
        new_vector = self.rng.uniform(size=dimensions)
        for i in range(dimensions):
            new_vector[i] = np.max(np.array(vector)[self.dim_map[i]])
        return new_vector

    def f_objective(self, x, fidelity=None, **kwargs):
        if self.f is None:
            raise NotImplementedError("An objective function needs to be passed.")
        if self.encoding:
            x = self.map_to_original(x)

        # Only convert config if configspace is used + configuration has not been converted yet
        if self.configspace:
            if not isinstance(x, ConfigSpace.Configuration):
                # converts [0, 1] vector to a ConfigSpace object
                config = self.vector_to_configspace(x)
            else:
                config = x
        else:
            config = x.copy()

        if fidelity is not None:  # to be used when called by multi-fidelity based optimizers
            res = self.f(config, fidelity=fidelity, **kwargs)
        else:
            res = self.f(config, **kwargs)
        assert "fitness" in res
        assert "cost" in res
        return res

    def init_eval_pop(self, fidelity=None, eval=True, **kwargs):
        '''Creates new population of 'pop_size' and evaluates individuals.
        '''
        self.population = self.init_population(self.pop_size)
        self.population_ids = self.config_repository.announce_population(self.population, fidelity)
        self.fitness = np.array([np.inf for i in range(self.pop_size)])
        self.age = np.array([self.max_age] * self.pop_size)

        traj = []
        runtime = []
        history = []

        if not eval:
            return traj, runtime, history

        for i in range(self.pop_size):
            config = self.population[i]
            config_id = self.population_ids[i]
            res = self.f_objective(config, fidelity, **kwargs)
            self.fitness[i], cost = res["fitness"], res["cost"]
            info = res["info"] if "info" in res else dict()
            if self.fitness[i] < self.inc_score:
                self.inc_score = self.fitness[i]
                self.inc_config = config
                self.inc_id = config_id
            self.config_repository.tell_result(config_id, float(fidelity or 0), res["fitness"], res["cost"], info)
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((config.tolist(), float(self.fitness[i]), float(fidelity or 0), info))

        return traj, runtime, history

    def eval_pop(self, population=None, population_ids=None, fidelity=None, **kwargs):
        '''Evaluates a population

        If population=None, the current population's fitness will be evaluated
        If population!=None, this population will be evaluated
        '''
        pop = self.population if population is None else population
        pop_ids = self.population_ids if population_ids is None else population_ids
        pop_size = self.pop_size if population is None else len(pop)
        traj = []
        runtime = []
        history = []
        fitnesses = []
        costs = []
        ages = []
        for i in range(pop_size):
            res = self.f_objective(pop[i], fidelity, **kwargs)
            fitness, cost = res["fitness"], res["cost"]
            info = res["info"] if "info" in res else dict()
            if population is None:
                self.fitness[i] = fitness
            if fitness <= self.inc_score:
                self.inc_score = fitness
                self.inc_config = pop[i]
                self.inc_id = pop_ids[i]
            self.config_repository.tell_result(pop_ids[i], float(fidelity or 0), info)
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((pop[i].tolist(), float(fitness), float(fidelity or 0), info))
            fitnesses.append(fitness)
            costs.append(cost)
            ages.append(self.max_age)
        if population is None:
            self.fitness = np.array(fitnesses)
            return traj, runtime, history
        else:
            return traj, runtime, history, np.array(fitnesses), np.array(ages)

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_rand2dir(self, r1, r2, r3):
        diff = r1 - r2 - r3
        mutant = r1 + self.mutation_factor * diff / 2
        return mutant

    def mutation(self, current=None, best=None, alt_pop=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop)
            mutant = self.mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5, alt_pop=alt_pop)
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == 'rand2dir':
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop)
            mutant = self.mutation_rand2dir(r1, r2, r3)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(size=2, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand1(best, r1, r2)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(size=4, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand2(best, r1, r2, r3, r4)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(size=2, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(current, best, r1, r2)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(r1, best, r2, r3)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.rng.random(self.dimensions) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.rng.integers(0, self.dimensions)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''Performs the exponential crossover of DE
        '''
        n = self.rng.integers(0, self.dimensions)
        L = 0
        while ((self.rng.random() < self.crossover_prob) and L < self.dimensions):
            idx = (n+L) % self.dimensions
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == 'exp':
            offspring = self.crossover_exp(target, mutant)
        return offspring

    def selection(self, trials, trial_ids, fidelity=None, **kwargs):
        '''Carries out a parent-offspring competition given a set of trial population
        '''
        traj = []
        runtime = []
        history = []
        for i in range(len(trials)):
            # evaluation of the newly created individuals
            res = self.f_objective(trials[i], fidelity, **kwargs)
            fitness, cost = res["fitness"], res["cost"]
            info = res["info"] if "info" in res else dict()
            # log result to config repo
            self.config_repository.tell_result(trial_ids[i], float(fidelity or 0), fitness, cost, info)
            # selection -- competition between parent[i] -- child[i]
            ## equality is important for landscape exploration
            if fitness <= self.fitness[i]:
                self.population[i] = trials[i]
                self.population_ids[i] = trial_ids[i]
                self.fitness[i] = fitness
                # resetting age since new individual in the population
                self.age[i] = self.max_age
            else:
                # decreasing age by 1 of parent who is better than offspring/trial
                self.age[i] -= 1
            # updation of global incumbent for trajectory
            if self.fitness[i] < self.inc_score:
                self.inc_score = self.fitness[i]
                self.inc_config = self.population[i]
                self.inc_id = self.population[i]
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((trials[i].tolist(), float(fitness), float(fidelity or 0), info))
        return traj, runtime, history

    def evolve_generation(self, fidelity=None, best=None, alt_pop=None, **kwargs):
        '''Performs a complete DE evolution: mutation -> crossover -> selection
        '''
        trials = []
        trial_ids = []
        for j in range(self.pop_size):
            target = self.population[j]
            donor = self.mutation(current=target, best=best, alt_pop=alt_pop)
            trial = self.crossover(target, donor)
            trial = self.boundary_check(trial)
            trial_id = self.config_repository.announce_config(trial, float(fidelity or 0))
            trials.append(trial)
            trial_ids.append(trial_id)
        trials = np.array(trials)
        trial_ids = np.array(trial_ids)
        traj, runtime, history = self.selection(trials, trial_ids, fidelity, **kwargs)
        return traj, runtime, history

    def sample_mutants(self, size, population=None):
        '''Generates 'size' mutants from the population using rand1
        '''
        if population is None:
            population = self.population
        elif len(population) < 3:
            population = np.vstack((self.population, population))

        old_strategy = self.mutation_strategy
        self.mutation_strategy = 'rand1'
        mutants = self.rng.uniform(low=0.0, high=1.0, size=(size, self.dimensions))
        for i in range(size):
            mutant = self.mutation(current=None, best=None, alt_pop=population)
            mutants[i] = self.boundary_check(mutant)
        self.mutation_strategy = old_strategy

        return mutants

    def run(self, generations=1, verbose=False, fidelity=None, reset=True, **kwargs):
        # checking if a run exists
        if not hasattr(self, 'traj') or reset:
            self.reset()
            if verbose:
                print("Initializing and evaluating new population...")
            self.traj, self.runtime, self.history = self.init_eval_pop(fidelity=fidelity, **kwargs)

        if verbose:
            print("Running evolutionary search...")
        for i in range(generations):
            if verbose:
                print("Generation {:<2}/{:<2} -- {:<0.7}".format(i+1, generations, self.inc_score))
            traj, runtime, history = self.evolve_generation(fidelity=fidelity, **kwargs)
            self.traj.extend(traj)
            self.runtime.extend(runtime)
            self.history.extend(history)

        if verbose:
            print("\nRun complete!")

        return np.array(self.traj), np.array(self.runtime), np.array(self.history, dtype=object)

    def ask(self, fidelity=None, **kwargs):
        """
        Generate a new candidate solution (offspring) to be evaluated externally.
        Returns a tuple (config, trial_id, target_idx) for use in tell().
        """
        assert len(self._ask_queue) == 0, "Previous ask() calls have not been told yet. Please call tell() before asking again."
        
        if not hasattr(self, 'population') or self.population is None:
            # If population is not initialized, initialize and evaluate it
            self.reset()
            self.init_eval_pop(fidelity=fidelity, eval=False, **kwargs)
            self.init = True
        
        # Select a target index (round-robin or random)
        if not hasattr(self, '_ask_counter'):
            self._ask_counter = 0
        
        if not hasattr(self, 'init'):
            self.init = False

        if self.init:
            # Round-robin selection for first pop_size calls
            target_idx = self._ask_counter % self.pop_size
            trial = self.population[target_idx]
            trial_id = self.population_ids[target_idx]
            self._ask_counter += 1
            if self._ask_counter >= self.pop_size:
                self.init = False
        else:
            if not hasattr(self, 'trials') or len(self.trials) == 0 or self.cur_trial_idx >= len(self.trials):
                self.trials = []
                self.trial_ids = []
                for j in range(self.pop_size):
                    target = self.population[j]
                    donor = self.mutation(current=target, best=self.inc_config)
                    trial = self.crossover(target, donor)
                    trial = self.boundary_check(trial)
                    trial_id = self.config_repository.announce_config(trial, float(fidelity or 0))
                    self.trials.append(trial)
                    self.trial_ids.append(trial_id)
                # selection takes place on a separate trial population only after
                # one iteration through the population has taken place
                self.trials = np.array(self.trials)
                self.cur_trial_idx = 0
            trial = self.trials[self.cur_trial_idx]
            trial_id = self.trial_ids[self.cur_trial_idx]
            target_idx = self.cur_trial_idx
            self.cur_trial_idx += 1

        if self.encoding:
            x = self.map_to_original(trial)
        else:
            x = trial

        # Only convert config if configspace is used + configuration has not been converted yet
        if self.configspace:
            if not isinstance(x, ConfigSpace.Configuration):
                # converts [0, 1] vector to a ConfigSpace object
                config = self.vector_to_configspace(x)
            else:
                config = x
        else:
            config = x.copy()
        return config, trial_id, target_idx

    def tell(self, trial, trial_id, target_idx, result, fidelity=None, **kwargs):
        """
        Accept a candidate and its evaluation result, updating the optimizer's state.
        result should be a dict with at least 'fitness' and 'cost'.
        """
        if self.configspace:
            trial = self.configspace_to_vector(trial)
        if self.encoding:
            trial = self.map_to_original(trial)
        
        fitness = result["fitness"]
        cost = result["cost"]
        info = result["info"] if "info" in result else dict()
        # Log result to config repo
        self.config_repository.tell_result(trial_id, float(fidelity or 0), fitness, cost, info)
        # Selection: replace parent if offspring is better or equal
        if fitness <= self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.population_ids[target_idx] = trial_id
            self.fitness[target_idx] = fitness
            self.age[target_idx] = self.max_age
        else:
            self.age[target_idx] -= 1
        # Update global incumbent: always scan the population for the best
        best_idx = np.argmin(self.fitness)
        self.inc_score = self.fitness[best_idx]
        self.inc_config = self.population[best_idx]
        self.inc_id = self.population_ids[best_idx]
        # Update trajectory/history
        if not hasattr(self, 'traj'):
            self.traj = []
        if not hasattr(self, 'runtime'):
            self.runtime = []
        if not hasattr(self, 'history'):
            self.history = []
        self.traj.append(self.inc_score)
        self.runtime.append(cost)
        self.history.append((trial.tolist(), float(fitness), float(fidelity or 0), info))
        return None
    
class AsyncDE(DE):
    def __init__(self, cs=None, f=None, dimensions=None, pop_size=None, max_age=np.inf,
                 mutation_factor=None, crossover_prob=None, strategy='rand1_bin',
                 async_strategy='immediate', seed=None, rng=None, config_repository=None, **kwargs):
        '''Extends DE to be Asynchronous with variations

        Parameters
        ----------
        async_strategy : str
            'deferred' - target will be chosen sequentially from the population
                the winner of the selection step will be included in the population only after
                the entire population has had a selection step in that generation
            'immediate' - target will be chosen sequentially from the population
                the winner of the selection step is included in the population right away
            'random' - target will be chosen randomly from the population for mutation-crossover
                the winner of the selection step is included in the population right away
            'worst' - the worst individual will be chosen as the target
                the winner of the selection step is included in the population right away
            {immediate, worst, random} implement Asynchronous-DE
        '''
        super().__init__(cs=cs, f=f, dimensions=dimensions, pop_size=pop_size, max_age=max_age,
                         mutation_factor=mutation_factor, crossover_prob=crossover_prob,
                         strategy=strategy, seed=seed, rng=rng, config_repository=config_repository,
                         **kwargs)
        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split('_')[0]
            self.crossover_strategy = self.strategy.split('_')[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None
        self.async_strategy = async_strategy
        assert self.async_strategy in ['immediate', 'random', 'worst', 'deferred'], \
                "{} is not a valid choice for type of DE".format(self.async_strategy)

    def _add_random_population(self, pop_size, population=None, fitness=[], age=[]):
        '''Adds random individuals to the population
        '''
        new_pop = self.init_population(pop_size=pop_size)
        new_fitness = np.array([np.inf] * pop_size)
        new_age = np.array([self.max_age] * pop_size)

        if population is None:
            population = self.population
            fitness = self.fitness
            age = self.age

        population = np.concatenate((population, new_pop))
        fitness = np.concatenate((fitness, new_fitness))
        age = np.concatenate((age, new_age))

        return population, fitness, age

    def _init_mutant_population(self, pop_size, population, target=None, best=None):
        '''Generates pop_size mutants from the passed population
        '''
        mutants = self.rng.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        for i in range(pop_size):
            mutants[i] = self.mutation(current=target, best=best, alt_pop=population)
        return mutants

    def _sample_population(self, size=3, alt_pop=None, target=None):
        '''Samples 'size' individuals for mutation step

        If alt_pop is None or a list/array of None, sample from own population
        Else sample from the specified alternate population
        '''
        population = None
        if isinstance(alt_pop, list) or isinstance(alt_pop, np.ndarray):
            idx = [indv is None for indv in alt_pop]  # checks if all individuals are valid
            if any(idx):
                # default to the object's initialized population
                population = self.population
            else:
                # choose the passed population
                population = alt_pop
        else:
            # default to the object's initialized population
            population = self.population

        if target is not None and len(population) > 1:
            # eliminating target from mutation sampling pool
            # the target individual should not be a part of the candidates for mutation
            for i, pop in enumerate(population):
                if all(target == pop):
                    population = np.concatenate((population[:i], population[i + 1:]))
                    break
        if len(population) < self._min_pop_size:
            # compensate if target was part of the population and deleted earlier
            filler = self._min_pop_size - len(population)
            new_pop = self.init_population(pop_size=filler)  # chosen in a uniformly random manner
            population = np.concatenate((population, new_pop))

        selection = self.rng.choice(np.arange(len(population)), size, replace=False)
        return population[selection]

    def eval_pop(self, population=None, population_ids=None, fidelity=None, **kwargs):
        pop = self.population if population is None else population
        pop_ids = self.population_ids if population_ids is None else population_ids
        pop_size = self.pop_size if population is None else len(pop)
        traj = []
        runtime = []
        history = []
        fitnesses = []
        costs = []
        ages = []
        for i in range(pop_size):
            res = self.f_objective(pop[i], fidelity, **kwargs)
            fitness, cost = res["fitness"], res["cost"]
            info = res["info"] if "info" in res else dict()
            if population is None:
                self.fitness[i] = fitness
            if fitness <= self.inc_score:
                self.inc_score = fitness
                self.inc_config = pop[i]
                self.inc_id = pop_ids[i]
            self.config_repository.tell_result(pop_ids[i], float(fidelity or 0), fitness, cost, info)
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((pop[i].tolist(), float(fitness), float(fidelity or 0), info))
            fitnesses.append(fitness)
            costs.append(cost)
            ages.append(self.max_age)
        return traj, runtime, history, np.array(fitnesses), np.array(ages)

    def mutation(self, current=None, best=None, alt_pop=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self._sample_population(size=3, alt_pop=alt_pop, target=current)
            mutant = self.mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self._sample_population(size=5, alt_pop=alt_pop, target=current)
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == 'rand2dir':
            r1, r2, r3 = self._sample_population(size=3, alt_pop=alt_pop, target=current)
            mutant = self.mutation_rand2dir(r1, r2, r3)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self._sample_population(size=2, alt_pop=alt_pop, target=current)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand1(best, r1, r2)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self._sample_population(size=4, alt_pop=alt_pop, target=current)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand2(best, r1, r2, r3, r4)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self._sample_population(size=2, alt_pop=alt_pop, target=current)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(current, best, r1, r2)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self._sample_population(size=3, alt_pop=alt_pop, target=current)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(r1, best, r2, r3)

        return mutant

    def sample_mutants(self, size, population=None):
        '''Samples 'size' mutants from the population
        '''
        if population is None:
            population = self.population

        mutants = self.rng.uniform(low=0.0, high=1.0, size=(size, self.dimensions))
        for i in range(size):
            j = self.rng.choice(np.arange(len(population)))
            mutant = self.mutation(current=population[j], best=self.inc_config, alt_pop=population)
            mutants[i] = self.boundary_check(mutant)

        return mutants

    def evolve_generation(self, fidelity=None, best=None, alt_pop=None, **kwargs):
        '''Performs a complete DE evolution, mutation -> crossover -> selection
        '''
        traj = []
        runtime = []
        history = []

        if self.async_strategy == "deferred":
            trials = []
            trial_ids = []
            for j in range(self.pop_size):
                target = self.population[j]
                donor = self.mutation(current=target, best=best, alt_pop=alt_pop)
                trial = self.crossover(target, donor)
                trial = self.boundary_check(trial)
                trial_id = self.config_repository.announce_config(trial, float(fidelity or 0))
                trials.append(trial)
                trial_ids.append(trial_id)
            # selection takes place on a separate trial population only after
            # one iteration through the population has taken place
            trials = np.array(trials)
            traj, runtime, history = self.selection(trials, trial_ids, fidelity, **kwargs)
            return traj, runtime, history

        elif self.async_strategy == "immediate":
            for i in range(self.pop_size):
                target = self.population[i]
                donor = self.mutation(current=target, best=best, alt_pop=alt_pop)
                trial = self.crossover(target, donor)
                trial = self.boundary_check(trial)
                trial_id = self.config_repository.announce_config(trial, float(fidelity or 0))
                # evaluating a single trial population for the i-th individual
                de_traj, de_runtime, de_history, fitnesses, costs = \
                    self.eval_pop(trial.reshape(1, self.dimensions),
                                  np.array([trial_id]), fidelity=fidelity, **kwargs)
                # one-vs-one selection
                ## can replace the i-the population despite not completing one iteration
                if fitnesses[0] <= self.fitness[i]:
                    self.population[i] = trial
                    self.population_ids[i] = trial_id
                    self.fitness[i] = fitnesses[0]
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)
            return traj, runtime, history

        else:  # async_strategy == 'random' or async_strategy == 'worst':
            for count in range(self.pop_size):
                # choosing target individual
                if self.async_strategy == "random":
                    i = self.rng.choice(np.arange(self.pop_size))
                else:  # async_strategy == 'worst'
                    i = np.argsort(-self.fitness)[0]
                target = self.population[i]
                mutant = self.mutation(current=target, best=best, alt_pop=alt_pop)
                trial = self.crossover(target, mutant)
                trial = self.boundary_check(trial)
                trial_id = self.config_repository.announce_config(trial, float(fidelity or 0))
                # evaluating a single trial population for the i-th individual
                de_traj, de_runtime, de_history, fitnesses, costs = \
                    self.eval_pop(trial.reshape(1, self.dimensions), np.array([trial_id]),
                                   fidelity=fidelity, **kwargs)
                # one-vs-one selection
                ## can replace the i-the population despite not completing one iteration
                if fitnesses[0] <= self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = fitnesses[0]
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)

        return traj, runtime, history

    

    def run(self, generations=1, verbose=False, fidelity=None, reset=True, **kwargs):
        # checking if a run exists
        if not hasattr(self, "traj") or reset:
            self.reset()
            if verbose:
                print("Initializing and evaluating new population...")
            self.traj, self.runtime, self.history = self.init_eval_pop(fidelity=fidelity, **kwargs)

        if verbose:
            print("Running evolutionary search...")
        for i in range(generations):
            if verbose:
                print("Generation {:<2}/{:<2} -- {:<0.7}".format(i+1, generations, self.inc_score))
            traj, runtime, history = self.evolve_generation(fidelity=fidelity,
                                                            best=self.inc_config, **kwargs)
            self.traj.extend(traj)
            self.runtime.extend(runtime)
            self.history.extend(history)

        if verbose:
            print("\nRun complete!")

        return np.array(self.traj), np.array(self.runtime), np.array(self.history, dtype=object)