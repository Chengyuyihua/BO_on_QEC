from bayesian_optimization.bo import BayesianOptimization
# from bayesian_optimization.acquisition_functions import AcquisitionFunction
from evaluation.decoder_based_evaluation import CSS_Evaluator
from bayesian_optimization.kernels import GNNKernel, WLSubtreeKernel,GNNEmbedding
from code_construction.code_construction import CodeConstructor
from bayesian_optimization.encoder import CSSEncoder
from bayesian_optimization.bo import ObjectiveFunction
import torch
from bayesian_optimization.gp import GaussianProcess_QEC
import matplotlib.pyplot as plt
import torch.nn as nn
from grakel.kernels import WeisfeilerLehman, VertexHistogram,RandomWalkLabeled
from gpytorch.kernels import ScaleKernel, RBFKernel
import numpy as np
DEVICE='cuda'

class Get_new_points_function():
    def __init__(self,method='qc-ldpc-hgp',hyperparameters = {'p': 2, 'q': 6, 'm': 2},encode='None'):
        self.method = method
        self.hyperparameters = hyperparameters
        self.encode = encode

    def get_new_points_function(self,number):
        if self.method == 'qc-ldpc-hgp':
            new_points = self.get_new_points_HGP(number)
        elif self.method == 'canonical_css':
            new_points = self.get_new_points_cananical(number,css = True)
        elif self.method == 'canonical':
            new_points = self.get_new_points_cananical(number)
        
        return new_points
        
    def get_new_points_HGP(self,number):
        return np.random.randint(0, self.hyperparameters['m'] + 1, (number, self.hyperparameters['p'] * self.hyperparameters['q']))
    def get_new_points_cananical(self,number,css = False):
        n = self.hyperparameters['n']
        r = self.hyperparameters['r']
        k = self.hyperparameters['k']
        if css == False:
            return np.random.randint(0,2,(number,(n-r)*r+k*(n-k)+(1+r)*r//2))
        else:
            return np.random.randint(0,2,(number,(n-r)*r+(n-k-r)*k))
class OverallEncode():
    def __init__(self,code_constructor ,encoding_method = 'graph',grakel_use = False):
        self.code_constructor = code_constructor
        self.encoding_method = encoding_method
        self.grakel_use = grakel_use
        self.encoder = CSSEncoder(code_constructor.n,code_constructor.nx,code_constructor.nz,mode = encoding_method,grakel_use=self.grakel_use)
    def encode(self,x):
        
        if self.grakel_use == False:
            result = torch.cat([self.encoder.encode(self.code_constructor.construct(np.array(i).astype(int))).unsqueeze(0) for i in x],dim=0)
            result.to(DEVICE)
            # print(f'{result.size()}')
        else:
            # print(x.size())
            if x.dim()==2:
                result = [self.encoder.encode(self.code_constructor.construct(np.array(i).astype(int))) for i in x]
            elif x.dim()==3:
                result = [self.encode(i) for i in x]
        
        return result




# WLSubtreeKernel(72,36,36,num_iterations=3,encoder=OverallEncode(codeconstructor,'graph').encode)

results = []
p = 2
q = 2
m = 2
para_dict = {'p':p,'q':q,'m':m}
codeconstructor = CodeConstructor(method='qc-ldpc-hgp',para_dict = {'p':p,'q':q,'m':m})
gnp = Get_new_points_function(method='qc-ldpc-hgp',hyperparameters = para_dict).get_new_points_function
LERPQ = ObjectiveFunction(codeconstructor,pp=0.05)

next_points_num = 4
candidate_num = 512
initial_sample_num = 15
bo_iteration = 100



gnp = Get_new_points_function(hyperparameters = {'p': p, 'q': p, 'm': m},encode='None').get_new_points_function
codeconstructor = CodeConstructor(method='qc-ldpc-hgp',para_dict = {'p':p,'q':q,'m':m})
encoder = OverallEncode(codeconstructor,encoding_method='graph',grakel_use=False)
gnnkernel_function = ScaleKernel(RBFKernel())
embedding = GNNEmbedding(m**2*(p**2+q**2),m**2*p*q,m**2*p*q)
bo = BayesianOptimization(
                object_function=LERPQ.lerpq,
                get_new_points_function= gnp,
                bounds= torch.tensor([[0.0 for i in range(p*q)], [m for i in range(p*q)]]),
                normalizer_mode='log_pos_trans',
                normalizer_positive = False,
                kernel =gnnkernel_function,
                encoder = encoder.encode,
                embedding=embedding,
                acquisition_function=None,
                BO_iterations= bo_iteration,
                initial_sample_num = initial_sample_num,
                candidate_num = candidate_num,
                next_points_num = next_points_num,
                training_num = 10,
                suggest_next_method = 'hill_climbing',
                description= 'Bayesian Optimization with Graph Neural Network kernel(with out pre-training) and qc-ldpc-hgp construction'
            )
para,ler,evaluation_history = bo.run()

plt.plot(evaluation_history)
plt.yscale('log')  
plt.xlabel('Iterations')
plt.ylabel('Evaluation History')
plt.title('Evaluation History (Log Scale)')
plt.savefig(f'./data/image/{bo.description}.png')

print(ler)
print(para)
results.append({'para':para,'ler':ler,'evaluation_history':evaluation_history,'method':bo.description})
print(results)