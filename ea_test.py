from evaluation.non_decoder_based_evaluation import Undetectable_error_rate
from code_construction.code_construction import CodeConstructor
import numpy as np
from evolutionary_algorithm.ea import CanonicalStabilizerEvolutionaryOptimization
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.algorithms.soo.nonconvex.ga import GA

from pymoo.optimize import minimize

# constructor = CodeConstructor(method='canonical',para_dict={'n':5,'k':1,'r':4})
# stabilizer_code = constructor.construct(np.array([1,1,1,1,1,0,0,1,0,0,1,0,0,1,1,0,0,0]))
# undetectable_error_rate = Undetectable_error_rate(stabilizer_code,noise_model=(1,1,1))
# print(undetectable_error_rate.evaluate(p=0.01))

problem = CanonicalStabilizerEvolutionaryOptimization({'n':5,'k':1,'r':4})        
algorithm = GA(
    pop_size=100,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True
)

res = minimize(problem,
            algorithm,
            termination=('n_gen', 200),
            seed=1,
)
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

