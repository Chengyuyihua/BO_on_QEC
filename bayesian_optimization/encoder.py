"""
    This file defines encoder that converts the input code into different representations
"""

import torch
from toponetx import CombinatorialComplex
from torch_geometric.utils import dense_to_sparse
from typing import List
from scipy.sparse import csr_matrix
from grakel import Graph
import numpy as np


class CSSEncoder():
    def __init__(self,n,nx,nz,mode='graph',grakel_use = False):
        """
            n: number of qubits, n = hx.shape[1] = hz.shape[1]
            nx: number of x stabilizers, nx = hx.shape[0]
            nz: number of z stabilizers, nz = hz.shape[0]
            mode: 'graph' or 'combinatorial_complex'
        """
        if mode == 'graph':
            self.encoder = GraphEncoder(n,nx,nz,grakel_use=grakel_use)
        elif mode == 'combinatorial_complex':
            self.encoder = CombinatorialComplexEncoder(n,nx,nz)
    def encode(self,c):
        hx = c.hx
        hz = c.hz
        return self.encoder.encode(hx,hz)
    
    


class GraphEncoder():
    def __init__(self,n,nx,nz,grakel_use=False):
        """
            n: number of qubits, n = hx.shape[1] = hz.shape[1]
            nx: number of x stabilizers, nx = hx.shape[0]
            nz: number of z stabilizers, nz = hz.shape[0]
        """
        self.n = n
        self.nx = nx
        self.nz = nz
        self.grakel_use = grakel_use
        
    def encode(self,hx,hz) -> torch.Tensor:
        adjacency_matrix = torch.zeros(self.n+self.nx+self.nz,self.n+self.nx+self.nz)
        for i in range(self.n):
            for j in range(self.nx):
                if hx[j][i] == 1:
                    adjacency_matrix[i][self.n+j] = 1
                    adjacency_matrix[self.n+j][i] = 1
            for j in range(self.nz):
                if hz[j][i] == 1:
                    adjacency_matrix[i][self.n+self.nx+j] = 1
                    # adjacency_matrix[self.n+self.nx+j][i] = 1
        # print(f'adjacency_matrix:{adjacency_matrix}')
        # edge_index = dense_to_sparse(adjacency_matrix)[0]
        # return edge_index
        if not self.grakel_use:
            return adjacency_matrix
        else:
            node_labels = {}
            for i in range(self.n):
                node_labels[i]='0'
            for i in range(self.n,self.n+self.nx):
                node_labels[i]='+'
            for i in range(self.n+self.nx,self.n+self.nx+self.nz):
                node_labels[i]='-'
            
            g = Graph(initialization_object=np.array(adjacency_matrix),node_labels=node_labels,graph_format='adjacency',construct_labels=False)
            # print(g)
            return g

    
class CombinatorialComplexEncoder():
    def __init__(self,n,nx,nz):
        """
            n: number of qubits, n = hx.shape[1] = hz.shape[1]
            nx: number of x stabilizers, nx = hx.shape[0]
            nz: number of z stabilizers, nz = hz.shape[0]
        """
        self.n_c0 = nx
        self.n_c1 = n
        self.n_c2 = nz
        
    def encode(self,hx,hz) -> List[csr_matrix]:
        """
            return Tensor:
            tensor([coA01,coA02,A10,coA12,A20,A21,B01,B02,B12])
        """
        CSS_cc = CombinatorialComplex()
        incidence_01 = [[] for _ in range(self.n_c1)]
        incidence_02 = [[] for _ in range(self.n_c2)]

        for i in range(self.n_c0): 
            for j in range(self.n_c1):  
                if hx[i][j] == 1:
                    incidence_01[j].append(i)

    
        for i in range(self.n_c2):  
            for j in range(self.n_c1):  
                if hz[i][j] == 1:
                    incidence_02[i].append(j) 

   
        for i in range(self.n_c1):
            CSS_cc.add_cell(incidence_01[i], rank=1)
        for i in range(self.n_c2):
            CSS_cc.add_cell(incidence_02[i], rank=2)
        
        return [
            torch.tensor(CSS_cc.coadjacency_matrix(1,0).todense()),
            torch.tensor(CSS_cc.coadjacency_matrix(2,0).todense()),
            torch.tensor(CSS_cc.adjacency_matrix(1,0).todense()),
            torch.tensor(CSS_cc.coadjacency_matrix(2,1).todense()),
            torch.tensor(CSS_cc.adjacency_matrix(2,0).todense()),
            torch.tensor(CSS_cc.adjacency_matrix(2,1).todense()),
            torch.tensor(CSS_cc.boundary_matrix(0,1).todense()),
            torch.tensor(CSS_cc.boundary_matrix(0,2).todense()),
            torch.tensor(CSS_cc.boundary_matrix(1,2).todense())
        ]
        # return [
        #     dense_to_sparse(torch.tensor(CSS_cc.coadjacency_matrix(1,0).todense()))[0],
        #     dense_to_sparse(torch.tensor(CSS_cc.coadjacency_matrix(2,0).todense()))[0],
        #     dense_to_sparse(torch.tensor(CSS_cc.adjacency_matrix(1,0).todense()))[0],
        #     dense_to_sparse(torch.tensor(CSS_cc.coadjacency_matrix(2,1).todense()))[0],
        #     dense_to_sparse(torch.tensor(CSS_cc.adjacency_matrix(2,0).todense()))[0],
        #     dense_to_sparse(torch.tensor(CSS_cc.adjacency_matrix(2,1).todense()))[0],
        #     dense_to_sparse(torch.tensor(CSS_cc.boundary_matrix(0,1).todense()))[0],
        #     dense_to_sparse(torch.tensor(CSS_cc.boundary_matrix(0,2).todense()))[0],
        #     dense_to_sparse(torch.tensor(CSS_cc.boundary_matrix(1,2).todense()))[0]
        # ]
    


if __name__ == '__main__':
    
    Hx = [[1,1,1,0,0,0,1,0],[0,0,1,0,1,1,1,0],[1,1,0,1,0,0,0,1],[0,0,0,1,1,1,0,1]]
    Hz = [[0,1,0,0,1,0,1,1],[0,1,1,1,1,0,0,0],[1,0,0,0,0,1,1,1],[1,0,1,1,0,1,0,0]]
    encoder = CSSEncoder(8,4,4,mode='graph',grakel_use=True)
    class c():
        def __init__(self,hx,hz):
            self.hx = hx
            self.hz=hz
    relations = encoder.encode(c(hx= Hx,hz= Hz))
    print(type(relations))
    # print(relations.dictionary)
    # print(relations)
    # print(type(relations[0]))

