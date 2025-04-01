"""
    This file contains the code constructions.

    The code constructions are the following:
        1.Stabilizer codes constructed by the canonical construction.
        2.CSS code constructed by the QC-LDPC-HGP construction.
        3.CSS code constructed by the Bivariate-bycicle codes.

"""


import numpy as np
class StabilizerCode():
    def __init__(self,h) -> None:
        self.h = h
        if not self.check_validity():
            raise ValueError('This stabilizer code is not valid.')
        self.n = self.h.shape[1]//2

    def check_validity(self):
        if not isinstance(self.h, np.ndarray):
            return False
        if self.h.shape[1]%2 != 0:
            return False
        n = self.h.shape[1]//2
        omega = np.block([
            [np.zeros((n,n)),np.eye(n)],
            [np.eye(n),np.zeros((n,n))]    
        ])
        if not ((self.h @ omega @ self.h.T) %2 == 0).all():
            return False

        return True
    
class CSSCode(StabilizerCode):
    def __init__(self,hx=None,hz=None) -> None:
        self.hx = hx
        self.hz = hz
        self.h = np.block([
            [hz, np.zeros(hz.shape)],
            [np.zeros(hx.shape), hx]
        ])
        if not self.check_validity():
            raise ValueError('This CSS code is not valid.')
        self.n = self.hx.shape[1]
        self.k = self.compute_css_k(hx,hz)
        
    
    def check_validity(self):
        if not isinstance(self.hx, np.ndarray)or not isinstance(self.hz, np.ndarray):
            print('not np.array')
            return False
        if self.hx.shape[1]!=self.hz.shape[1]:
            print('n is not equal')
            return False
        if ((self.hz @ self.hx.T%2) != 0).any():
            print('hz hx is not commute')
            return False
        return True
    def gf2_rank(self,A):
        '''
            mod2 rank
        '''

        A = A.copy() % 2
        m, n = A.shape
        rank = 0
        for col in range(n):
            pivot_row = None
            for row in range(rank, m):
                if A[row, col] == 1:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue 

            A[[rank, pivot_row]] = A[[pivot_row, rank]]

            for row in range(m):
                if row != rank and A[row, col] == 1:
                    A[row] = (A[row] + A[rank]) % 2
            rank += 1
            if rank == m:
                break
        return rank
    def compute_css_k(self,Hx, Hz):

        n = Hx.shape[1]
        r_hx = self.gf2_rank(Hx)
        r_hz = self.gf2_rank(Hz)
    
        k = n - r_hx - r_hz
        return k
    

class CodeConstructor():
    """
        This is the class for code construction.
        now we support the following methods:
            1. Canonical construction.(for Evolutionary algorithm, general stabilizer codes)
            2. QC-LDPC-HGP construction.
            3. Bivariate-bycicle construction.
    """
    def __init__(self,method='qc-ldpc-hgp',para_dict=None) -> None:
        """
            mothod(str): The method of the code construction, 'canonical', 'qc-ldpc-hgp', 'bivariate-bycicle'.
        """
        self.method = method
        self.para_dict = para_dict
        self.n = 0
        self.nx = 0
        self.nz = 0
        self.k = None
        
        if not self.check_parameters_validity():
            raise ValueError('The parameters are not valid.')
        
    def check_parameters_validity(self):
        if self.method == 'canonical' or self.method == 'canonical_css':
            return self.check_canonical_parameters_validity()
        elif self.method == 'qc-ldpc-hgp':
            return self.check_qc_ldpc_hgp_parameters_validity()
        elif self.method == 'bivariate-bycicle':
            return self.check_bivariate_bycicle_parameters_validity()  
        else:
            raise ValueError('The method is not supported.')

    def check_canonical_parameters_validity(self):
        """
            For each element in canonical construction requires the following parameters:
                1. n(int): The number of qubits.
                2. k(int): The number of logical qubits.
                3. r(int): The number of stabilizer generators at least have one x-operator.
                and s+r = n-k
        """
        for index in self.para_dict.keys():
            if type(self.para_dict[index])!=int or self.para_dict[index]<=0:
                return False
            if not index in {'n','k','r'}:
                return False
        if self.para_dict['n']-self.para_dict['k'] < self.para_dict['r']:
            return False
        self.n = self.para_dict['n']
        self.k = self.para_dict['k']
        self.nx = self.para_dict['r']
        self.nz = self.n-self.k-self.nx
        return True
    
    def check_qc_ldpc_hgp_parameters_validity(self):
        """
            For each element in QC-LDPC-HGP construction requires the following parameters:
                1. p(int): The number of rows of the matrix M.
                2. q(int): The number of columns of the matrix M.
                3. m(int): The size of the quasi-cyclic matrix.
                4*. p_2(int): The number of rows of the matrix M_2.  (* means optional)
                5*. q_2(int): The number of columns of the matrix M_2.
                6*. m_2(int): The size of the quasi-cyclic matrix M_2.
        """
        for index in self.para_dict.keys():
            if type(self.para_dict[index])!=int or self.para_dict[index]<=0:
                return False
        for i in  {'m','p','q'}:
            if not i in self.para_dict.keys():
                return False
        if 'p_2' in self.para_dict.keys():
            if not 'q_2' in self.para_dict.keys():
                return False
            if not 'm_2' in self.para_dict.keys():
                return False
            self.n = self.para_dict['m']*self.para_dict['m_2']*(self.para_dict['p']*self.para_dict['p_2']+self.para_dict['q']*self.para_dict['q_2'])
            self.nx = self.para_dict['m']*self.para_dict['m_2']*self.para_dict['p']*self.para_dict['q_2']
            self.nz = self.para_dict['m']*self.para_dict['m_2']*self.para_dict['p_2']*self.para_dict['q']
        else:
            self.n = self.para_dict['m']**2*(self.para_dict['p']**2 + self.para_dict['q']**2)
            self.nx = self.para_dict['m']**2*self.para_dict['p']*self.para_dict['q']
            self.nz = self.nx
        
 
        
        return True
    
    def check_bivariate_bycicle_parameters_validity(self):
        return False


    def construct(self,parameters):
        if self.method == 'canonical':
            return self.canonical_construction(parameters)
        elif self.method == 'canonical_css':
            return self.canonical_construction(parameters,CSS=True)
        elif self.method == 'qc-ldpc-hgp':
            return self.qc_ldpc_hgp_construction(parameters)
        elif self.method == 'bivariate-bycicle':
            return self.bivariate_bycicle_construction(parameters)
        else:
            raise ValueError('The method is not supported.')
    
    def canonical_construction(self,bitstring,CSS=False) -> StabilizerCode:
        """
            Construct the stabilizer code by the canonical construction.

            Args:
                parameters(np.array): The parameters of the code.

            Returns:
                stabilizer_code(StabilizerCode): The stabilizer code.
        """
        
        n = self.n
        k = self.k
        r = self.para_dict['r']
        s = n-k-r
        
        if CSS == False:
            if len(bitstring)!=((n-r)*r+k*(n-k)+(1+r)*r//2):
                raise ValueError('The length of the bitstring is invalid.')
            for i in bitstring:
                if i!=0 and i!=1:
                    raise ValueError("The bitstring's value is invalid.")
            A = bitstring[:r*(n-r)].reshape((r,n-r))
            A_1 = A[:,:s]
            A_2 = A[:,s:]
            C = bitstring[r*(n-r):r*(n-r)+k*(n-k)].reshape((n-k,k))
            C_1 = C[:r]
            C_2 = C[r:]
            M = np.zeros((r,r))
            # Reconstruct the symmetric matrix M
            p=0
            for i in range(r):
                for j in range(i,r):
                    
                    M[i,j] = bitstring[r*(n-r)+k*(n-k)+p]
                    M[j,i] = M[i,j]
                    p += 1
            if s!= 0:
                D = (A_1.T + C_2@A_2.T)%2
                B = (C_1@A_2.T + M)%2

                H = np.block([
                    [np.eye(r), A_1, A_2, B, np.zeros((r,s)),C_1],
                    [np.zeros((s,r)), np.zeros((s,s)),np.zeros((s,k)),D,np.eye(s),C_2]
                ])
            else:
                

                B = (C_1@A_2.T + M)%2

                H = np.block([
                    np.eye(r), A_2, B,C_1
                ])

            return StabilizerCode(H)
        else:
            if len(bitstring)!=((n-r)*r+s*k):
                raise ValueError('The length of the bitstring is invalid.')
            for i in bitstring:
                if i!=0 and i!=1:
                    raise ValueError("The bitstring's value is invalid.")
            A = bitstring[:r*(n-r)].reshape((r,n-r))
            A_1 = A[:,:s]
            A_2 = A[:,s:]
            C = bitstring[r*(n-r):r*(n-r)+k*(n-k)].reshape((s,k))
            D = (A_1.T + C@A_2.T)%2
            hx = np.block([
                D, np.eye(s), C
            ])
            hz = np.block([
                np.eye(r), A_1, A_2
            ])
            return CSSCode(hx,hz)



    def qc_ldpc_hgp_construction(self,M,M_2 = None ,form = None) -> CSSCode:
        """
            Construct the CSS code by the QC-LDPC-HGP construction.

            Args:
                M(np.ndarray): The parameters of the code.M is a np.ndarray
                M_2*(np.ndarray): The parameters of the code.M_2 is a np.ndarray

            Returns:
                CSS_code(CSSCode): The CSS code.
        """
        if M_2 is None:
            H1 = self.ldpc_construction(self.para_dict['p'],self.para_dict['q'],self.para_dict['m'],M)
            H2 = self.ldpc_construction(self.para_dict['p'],self.para_dict['q'],self.para_dict['m'],M)
        else:
            H1 = self.ldpc_construction(self.para_dict['p'],self.para_dict['q'],self.para_dict['m'],M)
            H2 = self.ldpc_construction(self.para_dict['p_2'],self.para_dict['q_2'],self.para_dict['m_2'],M_2)
        r1,n1 = H1.shape
        r2,n2 = H2.shape
        # HX = [H1 ⊗ In2 | Ir1 ⊗ H2.T]
        # HZ = [In1 ⊗ H2 | H1.T ⊗ Ir2]
        HX_left = np.kron(H1, np.eye(n2))
        HX_right = np.kron(np.eye(r1), H2.T)
        HX = np.hstack((HX_left, HX_right))

        HZ_left = np.kron(np.eye(n1), H2)
        HZ_right = np.kron(H1.T, np.eye(r2))
        HZ = np.hstack((HZ_left, HZ_right))

        
        return CSSCode(HX,HZ)
  
    def ldpc_construction(self,p,q,m,M):
        # print(M)
        if len(M)!=p*q:
            raise ValueError('The shape of M is invalid!')
        M = M.reshape(p,q)
        H = np.zeros((p*m,q*m))

        # Define the base cyclic shift matrix S (m x m)
        S = np.zeros((m, m))
        for i in range(m - 1):
            S[i, i + 1] = 1
        S[m - 1, 0] = 1
        # Construct H using the information from M
        for i in range(p):
            for j in range(q):
                # Get the value from matrix M
                shift = M[i, j] % (m+1)
                
                # Create the shifted version of S based on the value in M
                if shift == 0:
                    H_ij = np.zeros((m,m))  # Zero matrix for shift 0
                else:
                    H_ij = np.linalg.matrix_power(S, shift)
                
                # Place H_ij in the corresponding block of H
                H[i * m: (i + 1) * m, j * m: (j + 1) * m] = H_ij
        return H

    def bivariate_bycicle_construction(self,parameters) -> CSSCode:
        """
            Construct the CSS code by the Bivariate-bycicle construction.

            Args:
                parameters: The parameters of the code.

            Returns:
                CSS_code: The CSS code.
        """
        pass


if __name__ == '__main__':
    bitstring = np.array([1,1,1,1,1,0,0,1,0,0,1,0,0,1,1,0,0,0])
    para_dict = {'n':5,'k':1,'r':4}
    constructor = CodeConstructor('canonical',para_dict)
    code = constructor.construct(bitstring)
    bitstring_2 = np.array([1,0,1,1,0,1,1,1])
    para_dict_2 = {'n':5,'k':1,'r':2}
    constructor_2 = CodeConstructor('canonical_css',para_dict_2)
    code_2 = constructor_2.construct(bitstring_2)
    print(code_2.h)