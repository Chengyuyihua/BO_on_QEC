"""
    This file defines the Gaussian Process model used in Bayesian Optimization
"""
from typing import Optional

from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from torch import Tensor


class GaussianProcess_QEC(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, kernel=None,encoder = None,embedding = None,train_Yvar: Optional[Tensor] = None):
        # NOTE: This ignores train_Yvar and uses inferred noise instead.
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        if encoder==None:
            def encode_(x):
                return x
            self.encode = encode_
        else:
            self.encode = encoder
        if embedding==None:
            def embed_(x):
                return x
            self.embed = embed_
        else:
            self.embed = embedding
      
        self.mean_module = ConstantMean()
        self.covar_module = kernel
        self.to(train_X)  

    def forward(self, x):
        mean_x = self.mean_module(x)
        embedding_x = self.embed(self.encode(x))
        # print(f'embedding_x:{type(embedding_x)},{embedding_x.size()}')
        covar_x = self.covar_module(embedding_x)
        
        # print(f'covar_x:{type(covar_x)},{covar_x}')
        
        return MultivariateNormal(mean_x, covar_x)