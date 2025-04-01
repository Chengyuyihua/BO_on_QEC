from bayesian_optimization.bo import BayesianOptimization
# from bayesian_optimization.acquisition_functions import AcquisitionFunction
from evaluation.decoder_based_evaluation import CSS_Evaluator
from bayesian_optimization.kernels import GNNKernel, WLSubtreeKernel,GNNEmbedding
from code_construction.code_construction import CodeConstructor
from bayesian_optimization.encoder import CSSEncoder
from bayesian_optimization.bo import LogicalErrorRatePerQubit
import torch
from bayesian_optimization.gp import GaussianProcess_QEC
import matplotlib.pyplot as plt
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
p = 4
q = 4
m = 4
para_dict = {'p':p,'q':q,'m':m}
codeconstructor = CodeConstructor(method='qc-ldpc-hgp',para_dict = {'p':p,'q':q,'m':m})
gnp = Get_new_points_function(method='qc-ldpc-hgp',hyperparameters = para_dict).get_new_points_function
LERPQ = LogicalErrorRatePerQubit(codeconstructor,pp=0.01)
# print(LERPQ.forward(gnp(3)[0]))

# n = 31
# k= 1
# r = 15
# para_dict = {'n':n,'k':k,'r':r}
# codeconstructor = CodeConstructor(method='canonical_css',para_dict = para_dict)
# gnp = Get_new_points_function(method='canonical_css',hyperparameters = para_dict).get_new_points_function
# # print(codeconstructor.construct(gnp(1)[0]).hx)
# 
# LERPQ = LogicalErrorRatePerQubit(codeconstructor,pp=0.01)
# print(LERPQ.forward(gnp(1)[0]))
next_points_num = 4
candidate_num = 512
initial_sample_num = 15
bo_iteration = 50

bo = BayesianOptimization(
                object_function=LERPQ.forward,
                get_new_points_function= gnp,
                bounds= torch.tensor([[0.0 for i in range(p*q)], [m for i in range(p*q)]]),
                normalizer_mode='log_pos_trans',
                kernel =ScaleKernel(RBFKernel()),
                encoder = None,
                acquisition_function=None,
                BO_iterations= bo_iteration,
                initial_sample_num = initial_sample_num,
                candidate_num = candidate_num,
                next_points_num = next_points_num,
                training_num = 5,
                description= f'Bayesian Optimization with RBF kernel and qc-ldpc-hgp construction,para_dict:{para_dict}'
            )

para,ler,evaluation_history = bo.run()

results.append({'para':para,'ler':ler,'evaluation_history':evaluation_history,'method':bo.description,'para_dict':para_dict})
plt.plot(evaluation_history)
plt.yscale('log')  
plt.xlabel('Iterations')
plt.ylabel('Evaluation History')
plt.title('Evaluation History (Log Scale)')
plt.savefig(f'./data/image/{bo.description}.png', dpi=300)
# plt.show()

print(ler)
print(para)
random_points = gnp(bo_iteration*next_points_num+initial_sample_num)



ler_random = [LERPQ.forward(x) for x in random_points]
print('random search results:')
print(random_points[np.argmin(ler_random)])
print(min(ler_random))
results.append({'para':random_points[np.argmin(ler_random)],'ler':min(ler_random),'evaluation_history':[],'method':'random search','para_dict':para_dict})
kernel = WeisfeilerLehman(n_iter=4, normalize=True, base_graph_kernel=VertexHistogram)
def make_positive_semidefinite(kernel_matrix, epsilon=1e-3):
    regularized_kernel = kernel_matrix + epsilon * torch.eye(kernel_matrix.size(0))
    eigenvalues, eigenvectors = torch.linalg.eigh(regularized_kernel)
    eigenvalues_clipped = torch.clamp(eigenvalues, min=0)
    kernel_matrix_psd = eigenvectors @ torch.diag(eigenvalues_clipped) @ eigenvectors.T
    return kernel_matrix_psd.float()
def WLkernel_function(x):
    return make_positive_semidefinite(torch.tensor(kernel.fit_transform(x)))
gnp = Get_new_points_function(hyperparameters = {'p': p, 'q': p, 'm': m},encode='None').get_new_points_function
encoder = OverallEncode(codeconstructor,encoding_method='graph',grakel_use=True)
bo = BayesianOptimization(
                object_function=LERPQ.forward,
                get_new_points_function= gnp,
                bounds= torch.tensor([[0.0 for i in range(p*q)], [m for i in range(p*q)]]),
                normalizer_mode='log_pos_trans',
                kernel =WLkernel_function,
                encoder = encoder.encode,
                acquisition_function=None,
                BO_iterations= bo_iteration,
                initial_sample_num = initial_sample_num,
                candidate_num = candidate_num,
                next_points_num = next_points_num,
                training_num = 1,
                description= f'Bayesian Optimization with WL kernel and qc-ldpc-hgp construction,para_dict:{para_dict}'
            )


para,ler,evaluation_history = bo.run()
results.append({'para':para,'ler':ler,'evaluation_history':evaluation_history,'method':bo.description,'para_dict':para_dict})
plt.plot(evaluation_history)
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Evaluation History')
plt.title('Evaluation History (Log Scale)')
plt.savefig(f'./data/image/{bo.description}.png', dpi=300)
# plt.show()

print(ler)
print(para)



# random walk kernel
def RWkernel_function(x):
    return make_positive_semidefinite(torch.tensor(RandomWalkLabeled(normalize=True).fit_transform(x)))

bo = BayesianOptimization(
                object_function=LERPQ.forward,
                get_new_points_function= gnp,
                bounds= torch.tensor([[0.0 for i in range(p*q)], [m for i in range(p*q)]]),
                normalizer_mode='log_pos_trans',
                kernel =RWkernel_function,
                encoder = encoder.encode,
                acquisition_function=None,
                BO_iterations= bo_iteration,
                initial_sample_num = initial_sample_num,
                candidate_num = candidate_num,
                next_points_num = next_points_num,
                training_num = 1,
                description= f'Bayesian Optimization with WL kernel and qc-ldpc-hgp construction,para_dict:{para_dict}'
            )



















gnp = Get_new_points_function(hyperparameters = {'p': p, 'q': p, 'm': m},encode='None').get_new_points_function
codeconstructor = CodeConstructor(method='qc-ldpc-hgp',para_dict = {'p':p,'q':q,'m':m})
encoder = OverallEncode(codeconstructor,encoding_method='graph',grakel_use=False)
gnnkernel_function = ScaleKernel(RBFKernel())
embedding = GNNEmbedding(m**2*(p**2+q**2),m**2*p*q,m**2*p*q)
bo = BayesianOptimization(
                object_function=LERPQ.forward,
                get_new_points_function= gnp,
                bounds= torch.tensor([[0.0 for i in range(p*q)], [m for i in range(p*q)]]),
                normalizer_mode='log_pos_trans',
                kernel =gnnkernel_function,
                encoder = encoder.encode,
                embedding=embedding,
                acquisition_function=None,
                BO_iterations= bo_iteration,
                initial_sample_num = initial_sample_num,
                candidate_num = candidate_num,
                next_points_num = next_points_num,
                training_num = 10,
                description= f'Bayesian Optimization with Graph Neural Network kernel(with out pre-training) and qc-ldpc-hgp construction,para_dict:{para_dict}'
            )
para,ler,evaluation_history = bo.run()

plt.plot(evaluation_history)
plt.yscale('log')  
plt.xlabel('Iterations')
plt.ylabel('Evaluation History')
plt.title('Evaluation History (Log Scale)')
plt.savefig(f'./data/image/{bo.description}.png', dpi=300)
# plt.show()

print(ler)
print(para)
results.append({'para':para,'ler':ler,'evaluation_history':evaluation_history,'method':bo.description,'para_dict':para_dict})