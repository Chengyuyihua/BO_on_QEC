"""
    This file contains the implementation of the Bayesian Optimization algorithm
"""


from typing import Callable, Tuple, List, Dict, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel as C
import numpy as np
import torch
from torch import Tensor
from numpy import ndarray
import pickle
import tqdm
import torch
import numpy as np
import botorch
from botorch.models import SingleTaskGP
from torch import nn
from botorch.optim import optimize_acqf
from botorch.acquisition import AcquisitionFunction
from gpytorch.mlls import ExactMarginalLogLikelihood
# from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from torch import Tensor
from torch.optim import Adam
import gpytorch
from torch.distributions.normal import Normal
import tqdm
from bayesian_optimization.gp import GaussianProcess_QEC
from evaluation.decoder_based_evaluation import CSS_Evaluator
from code_construction.code_construction import CodeConstructor
import time
from scipy.special import gammaln
from scipy.optimize import root_scalar

# class Batch_Bayesian_Optimization():
    # def __init__(self):
    #     pass
    # def evaluate_batch(self,object_function: Callable,x_batch: Tensor)->Tensor:
    #     '''
    #         return the value of the object function of this batch
    #     '''
    #     pass
    # def suggest_next_batch(self) -> Tensor:
    #     pass
    # def run_BO(self,object_function: Callable,initial_points: Tensor,initial_values: Tensor)->Tuple[Tensor,Tensor]:
    #     pass
    # def load_BO():
    #     pass
    # def save_BO():
    #     pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LogicalErrorRatePerQubit():
    """
    The lower the better
    """
    def __init__(self,code_constructor: CodeConstructor,pp=0.01):
        self.code_constructor = code_constructor
        self.pp = pp
    def LERforward(self,css):
        
        evaluator = CSS_Evaluator(css.hx,css.hz)
        
        # pL = evaluator.Get_error_rate(physical_error_rate=self.pp)
        pL,_ = evaluator.Get_precise_logical_error_rate(physical_error_rate=self.pp,trail=300,block = 204)
        
        
        return pL
    def forward(self,x):
        css = self.code_constructor.construct(x)
        if css.k==0:
            return 1
        pL = self.LERforward(css)
        pL_per_lq = 1-(1-pL)**(1/css.k)
        return pL_per_lq
    
    def psuedo_distance(self,n,pL):
        c1 = gammaln(n+1)
        c2 = np.log(pL)
        c3 = np.log(self.pp)
        def F(t,n):
            return c1 - gammaln(t+1) - gammaln(n-t+1) + (t+1)*c3 - c2
        sol = root_scalar(F, args=(n), bracket=[0, n//2], method='brentq', xtol=1e-4)
        t_test = sol.root
        return 2*t_test
    def distance(self,x):
        css = self.code_constructor.construct(x)
        if css.k==0:
            return 1
        pL = self.LERforward(css)
        return self.psuedo_distance(css.n,pL)

class Normalizer:
    def __init__(self, mode='log_pos_trans', epsilon=1e-11):
        self.epsilon = epsilon
        self.mean = 0
        self.std = 1
        self.mode = mode

    def normalize(self, x):

        if self.mode == 'log_pos_trans':
            return -self.log_pos_trans(x)

    def get_mean_std(self,X):
        if self.mode == 'log_pos_trans':
            log_outputs = torch.tensor([self.log_pos_trans(i) for i in X])
            self.mean = log_outputs.mean()
            self.std = log_outputs.std()
        return self.mean, self.std

    def inverse_normalize(self, y):
        if self.mode == 'log_pos_trans':
            return self.inverse_log_pos_trans(-y)

    def log_pos_trans(self, x):
        y = x / (1 - x + self.epsilon)
        return (torch.log(y) - self.mean)/self.std

    def inverse_log_pos_trans(self, y):
        x = torch.exp(self.std * y + self.mean)
        return x / (1 + x)
    


def train_gpytorch_model(model, train_x, train_y, training_iter=50, lr=0.002):
    model.train()
    model.likelihood.train()
    
    optimizer = Adam(model.parameters(), lr=lr)
    

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        # if i % 10 == 0:
            # print(f'Iteration {i+1}/{training_iter} - Loss: {loss.item()}')
        
    return loss.item()

class BayesianOptimization():# for single point optimization; using botorch
    def __init__(self,
                    object_function: Callable,
                    get_new_points_function: Callable,
                    bounds: Tensor,
                    normalizer_mode='log_pos_trans',
                    kernel = None,
                    encoder = None,
                    embedding = None,
                    acquisition_function: Callable = None,
                    BO_iterations: int = 10,
                    initial_k_detection=True,
                    initial_sample_num = 15,
                    candidate_num = 512,
                    next_points_num = 10,
                    training_num = 20,
                    description: str = 'Bayesian Optimization'
                ):
        '''
            Initialize the Bayesian Optimization
        '''
        self.BO_iterations = BO_iterations
        self.description = description
        # initialize the parameters
        self.bounds = bounds
        self.Normalizer = Normalizer(mode=normalizer_mode)

        self.object_function = object_function
        self.get_new_points_function = get_new_points_function
        # we don't want initial X to have too many k=0 points
        
        
        if initial_k_detection:
            self.X,y_not_normalized = self.update_points_until_valid(initial_sample_num)
        else:
            self.X = self.get_new_points_function(initial_sample_num)
            # print(f'Initializing the object function values of the initial points...{self.X}')
            y_not_normalized = torch.tensor([self.object_function(x) for x in self.X])
        self.X = torch.tensor(self.X,dtype=torch.float32)
        self.X.to(DEVICE)
        self.Normalizer.get_mean_std(y_not_normalized)

        print(f'Initializing the object function values of the initial points...{y_not_normalized}')
        self.y = torch.tensor([self.Normalizer.normalize(y) for y in y_not_normalized],dtype=torch.float32)
        self.y.to(DEVICE)
        print(f'The performance of the initial points:{type(self.X)},{self.X.size()},{type(self.y)},{self.y.size()},{self.y}')

        # initialize the GP model
        if kernel is None:
            kernel = ScaleKernel(RBFKernel(ard_num_dims=self.X.shape[-1]))
        
        if encoder is None:
            def encode_(x):
                return x
            self.encoder = encode_
        else:
            self.encoder=encoder
        
  
        self.gp = GaussianProcess_QEC(self.X, self.y,kernel=kernel,encoder=self.encoder,embedding=embedding)
        # fit the GP model
        print('fitting the GP model...')
        self.training_num = training_num
        train_gpytorch_model(self.gp, self.X, self.y, training_iter=training_num*2, lr=0.05)
        # Check the fit of the GP model


        # initialize the best value
        self.best_value = self.y.max()
        self.best_parameters = self.X[self.y.argmax()]

        print(f'Initial best value: {self.best_value:.4f}')

        # initialize the acquisition function
        
        if acquisition_function is None:
            self.bv = self.best_value
            self.acquisition_function = botorch.acquisition.LogExpectedImprovement(self.gp, self.bv)
        else:
            self.acquisition_function = acquisition_function
        self.num_candidates = candidate_num
        self.next_points_num = next_points_num
    def update_points_until_valid(self, num_points=15):

        X = self.get_new_points_function(num_points)
        y_not_normalized = torch.tensor([self.object_function(x) for x in X])
        

        def get_duplicate_indices(X):
            points_list = [tuple(x.tolist()) if isinstance(x, torch.Tensor) else tuple(x) for x in X]
            seen = {}
            duplicate_idx = []
            for i, p in enumerate(points_list):
                if p in seen:
                    duplicate_idx.append(i)
                else:
                    seen[p] = i
            return duplicate_idx


        while True:

            indices_y_eq_1 = [i for i, y in enumerate(y_not_normalized) if y == 1]

            duplicate_indices = get_duplicate_indices(X)

            indices_to_replace = set(indices_y_eq_1 + duplicate_indices)
            
            if not indices_to_replace:

                break
            

            indices_to_replace = sorted(list(indices_to_replace))  # 排序保证顺序一致
            new_points = self.get_new_points_function(len(indices_to_replace))

            for idx, new_point in zip(indices_to_replace, new_points):
                X[idx] = new_point
                y_not_normalized[idx] = self.object_function(new_point)
        
        return X, y_not_normalized




        
    def evaluate(self,x: Tensor)->Tensor:
        '''
            return the value of the object function of this batch
        '''
       
        return torch.tensor([self.Normalizer.normalize(torch.tensor(self.object_function(i))) for i in x],dtype=torch.float32)
    # def suggest_next(self) -> Tensor:
    #     '''
    #         return the next point to evaluate
    #     '''

    #     X_0 = self.get_new_points_function(512)
    #     X = torch.tensor(X_0, dtype=torch.float32)
    #     print(self.acquisition_function(torch.cat(self.X,X[0].unsqueeze(0),dim=0)))
    #     X = X.unsqueeze(1)
    #     # X.to('cuda')
    #     # print(f'X:{type(X)},{X.size()}')
    #     estimate = self.acquisition_function(X)  
    #     best_index = torch.argmax(estimate)
    #     best_candidate = X_0[best_index]
    #     # print(f'best_candidate:{best_candidate}')
    #     return [best_candidate]
    # def suggest_next(self) -> Tensor:
    #     '''
    #         return the next point to evaluate
    #     '''

    #     X_0 = self.get_new_points_function(512)
    #     X = torch.tensor(X_0, dtype=torch.float32)


    #     estimates = []

    #     self.gp.eval()
    #     self.gp.likelihood.eval()
    #     for x in X:

    #         x = x.unsqueeze(0)
    #         # print(x.size())
            
    #         with torch.no_grad(), gpytorch.settings.fast_pred_var():

    #             observed_pred = self.gp(x)
    #             mean_x = observed_pred.mean
    #             std_dev_x = observed_pred.stddev
    #             normal = Normal(torch.zeros_like(mean_x), torch.ones_like(std_dev_x))
    #             imp = mean_x - self.best_value - 0.01
    #             Z = imp / std_dev_x
    #             ei = imp * normal.cdf(Z) + std_dev_x * normal.log_prob(Z).exp()

    #         estimates.append(ei)


    #     estimates_tensor = torch.cat(estimates)

    #     best_index = torch.argmax(estimates_tensor)
    #     best_candidate = X_0[best_index]

    #     return [best_candidate]
    def suggest_next(self) -> Tensor:
        '''
            return the next point to evaluate
        '''

        X_0 = self.get_new_points_function(self.num_candidates)
        X = torch.tensor(X_0, dtype=torch.float32)
        X.to(DEVICE)


        # estimates = []

        self.gp.eval()
        self.gp.likelihood.eval()
            
        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            observed_pred = self.gp(X)
            mean_x = observed_pred.mean
            std_dev_x = observed_pred.stddev
            normal = Normal(torch.zeros_like(mean_x), torch.ones_like(std_dev_x))
            imp = mean_x - self.best_value - 0.01
            Z = imp / std_dev_x
            ei = imp * normal.cdf(Z) + std_dev_x * normal.log_prob(Z).exp()


        values, indices = torch.topk(ei, self.next_points_num)
        best_candidates = X_0[indices]

        return best_candidates
    
    def run(self)->Tuple[Tensor,Tensor]:
        '''
            Run the Bayesian Optimization
        '''
        evaluation_history= []
        best_y_paras = []
        pbar = tqdm.tqdm(range(self.BO_iterations), desc=self.description)
        for i in pbar:
            # Step 1: Find the next point to evaluate by maximizing the EI acquisition function
            t0 = time.time()
            next_points = self.suggest_next()
            t_next = time.time()

            # Step 2: Obtain the value of the object function
            next_values = self.evaluate(next_points)
            t_evaluated = time.time()
            evaluation_history.append(self.Normalizer.inverse_normalize(next_values))
            # print(f'nextpoint:{type(next_point)}')
            # Step 3: Update the GP model
            # print(type(self.X))
            self.X = torch.cat((self.X, torch.tensor(next_points)), dim=0)
            self.y = torch.cat((self.y, next_values), dim=0)
            # print(f'self.X:{type(self.X)}')
            # print(self.y)
            t1 = time.time()
            self.gp.set_train_data(inputs=self.X, targets=self.y, strict=False)
            
            loss = train_gpytorch_model(self.gp,self.X, self.y, training_iter=self.training_num, lr=0.01)
            t_train = time.time()
            # Step 4: Update the best value
            next_value_best = next_values.max()
            next_point_best = next_points[next_values.argmax()]
            if next_value_best > self.best_value:
                self.best_value = next_value_best
                self.best_parameters = next_point_best
                best_y_paras.append([self.best_value,self.best_parameters])
            # Update progress bar description with current best value
            pbar.set_description(f'{self.description} (Best value: {self.Normalizer.inverse_normalize(self.best_value)})(GP MLL: {loss}),time:suggesting next:{t_next-t0},evaluate: {t_evaluated-t_next},training: {t_train-t1}')
        # print(self.best_parameters)
        
        return self.best_parameters,self.Normalizer.inverse_normalize(self.best_value),evaluation_history

    def load(self, filename: str) -> None:
        """
        Load the BO model from a file
        """
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            self.X = state['X']
            self.y = state['y']
            self.gp = state['gp']
            self.best_value = state['best_value']
            self.best_parameters = state['best_parameters']

    def save(self, filename: str) -> None:
        """
        Save the BO model to a file
        """
        state = {
            'X': self.X,
            'y': self.y,
            'gp': self.gp,
            'best_value': self.best_value,
            'best_parameters': self.best_parameters
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)


if __name__ == '__main__':
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    def object_function(x):
        return torch.sin(x[0])**2 * torch.cos(x[1])**2

    def get_new_points_function(n_points):
        return torch.rand(n_points, 2) * (bounds[1] - bounds[0]) + bounds[0]

    

    bo = BayesianOptimization(object_function, get_new_points_function, bounds)
    best_parameters, best_value = bo.run()
    print(f"Best parameters: {best_parameters}")
    print(f"Best value: {best_value}")