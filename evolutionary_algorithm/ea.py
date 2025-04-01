"""
    This file contains the implementation of the Evolutionary Algorithm on the QEC codes.
"""
from pymoo.core.problem import Problem
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

from code_construction.code_construction import CodeConstructor
from evaluation.non_decoder_based_evaluation import Undetectable_error_rate

class CanonicalStabilizerEvolutionaryOptimization(Problem):
    def __init__(self,paradict = {'n':0,'k':0,'r':0},p = 0.01,noise_model=(1,1,1)):
        n = paradict['n']
        k = paradict['k']
        r = paradict['r']
        self.p = p
        self.noise_model = noise_model
        nbits = (n-r)*r+k*(n-k)+(1+r)*r//2
        self.constructor = CodeConstructor(method='canonical',para_dict=paradict)
        super().__init__(n_var=nbits,         
                         n_obj=1,         
                         n_constr=0,
                         xl=np.zeros(nbits),  
                         xu=np.ones(nbits),   
                         vtype=int)     
        

    def _evaluate(self, x, out, *args, **kwargs):
        # print(len(x[0]))
        out_list = []
        for i in x:
            stabilizer_code = self.constructor.construct(i)
            evaluator = Undetectable_error_rate(stabilizer_code,noise_model=self.noise_model)
            out_list.append(evaluator.evaluate(self.p))
        out["F"] = np.array(out_list)
class CanonicalCSSEvolutionaryOptimization(Problem):
    def __init__(self,nbits):
        super().__init__(n_var=nbits,         
                         n_obj=1,         
                         n_constr=0,
                         xl=np.zeros(nbits),  
                         xu=np.ones(nbits),   
                         type_var=int)     
                            

    def _evaluate(self, x, out, *args, **kwargs):
        # calculate the ebaluation function
        f = np.sum(x**2, axis=1)
        out["F"] = f
if __name__ == '__main__':
    problem = CanonicalStabilizerEvolutionaryOptimization({'n':5,'k':1,'r':4})        
    algorithm = GA(
        pop_size=100,
        sampling=BinaryRandomSampling(),
        crossover=HUXCrossover(prob=0.9),
        mutation=BinaryBitflipMutation(prob=0.1)
    )

    res = minimize(problem,
               algorithm,
               termination=('n_gen', 200),
               seed=1,
    )