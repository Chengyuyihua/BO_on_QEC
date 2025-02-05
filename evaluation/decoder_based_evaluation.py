"""
This module contains the functions to evaluate the performance of the code with a decoder.
"""

import numpy as np
from css_decode_sim import css_decode_sim

from bposd.hgp import hgp
from ldpc.codes import rep_code
from scipy.linalg import null_space
from scipy.special import comb
from itertools import product
from math import floor
import matplotlib.pyplot as plt
from bposd import bposd_decoder
from bposd.css import css_code
def binomial_probability(n, d_e,p_p):
    return comb(n, d_e) * (p_p ** d_e) * ((1 - p_p) ** (n - d_e))
def gf2_rref(H):
    """ Transform the matrix H to row echelon form """
    H = H.copy()
    rows, cols = H.shape
    r = 0  # The row we're working on
    for c in range(cols):
        # Find the pivot row and swap it to the top
        pivot = np.where(H[r:, c] == 1)[0]
        if pivot.size == 0:
            continue  # No pivot in this column, skip it
        pivot_row = pivot[0] + r
        H[[r, pivot_row]] = H[[pivot_row, r]]
        
        # Zero out all other entries in this column
        for ri in range(rows):
            if ri != r and H[ri, c] == 1:
                H[ri] ^= H[r]
        
        r += 1
        if r == rows:
            break
    
    return H

def find_null_space(H):
    """ calculate the null space of the matrix H """
    # Transform H to row echelon form
    rref = gf2_rref(H)
    rows, cols = rref.shape
    pivot_cols = np.array([np.where(rref[r])[0][0] for r in range(rows) if np.any(rref[r])])
    free_cols = np.array([c for c in range(cols) if c not in pivot_cols])
    
    # Generate the matrix G
    G = np.zeros((len(free_cols), cols), dtype=int)
    
    # Set the identity matrix in the free columns
    for i, f_col in enumerate(free_cols):
        G[i, f_col] = 1  # set the free variable
        for r, p_col in enumerate(pivot_cols):
            if p_col > f_col:
                break
            if rref[r, f_col] == 1:
                G[i, p_col] = 1  # set the pivot variable
    
    return G

def Get_G_of_check_matrix(H):
    G = find_null_space(H)
    return G.T
def Get_distribution_dij(G,trails=1000):
    d_ij={}
    for i in range(trails):
        x_i = np.random.randint(0,2,(G.shape[1],1))
        x_j = np.random.randint(0,2,(G.shape[1],1))
        if (x_i == x_j).all():
            continue
        else:
            
            
            d = (G@x_i%2 + G@x_j%2)%2
            d = np.sum(d)
            if d in d_ij:
                d_ij[d] += 1
            else:
                d_ij[d] = 1
    return d_ij


class CSS_Evaluator():
    def __init__(self,hx,hz):
        self.hx = hx
        self.hz = hz
        qcode = css_code(self.hx, self.hz)
        self.lx = qcode.lx
        self.lz = qcode.lz
        self.k = qcode.K
        self.n = qcode.N
        

    def Get_error_rate(self,physical_error_rate=0.05,trail=1000):
        if self.k == 0:
            return 1
        decoder_sim = css_decode_sim(
            hx=self.hx,
            hz=self.hz,
            error_rate= physical_error_rate,
            xyz_error_bias= [1,1, 1],
            target_runs= trail,
            seed= 0,
            bp_method= "minimum_sum",
            ms_scaling_factor= 0.625,
            max_iter= 10,
            osd_method= "osd_cs",
            osd_order= 3,
            save_interval= 2,
            output_file= None,
            check_code= 0,
            tqdm_disable= 0,
            run_sim= 0,
            channel_update= None,
            hadamard_rotate= 0,
            hadamard_rotate_sector1_length= 0,
            error_bar_precision_cutoff= 1e-4
        )
        logical_error_rate = decoder_sim.run_decode_sim()

        return logical_error_rate

    def init_decoder(self,physical_error_rate):
        self.bpd_z = bposd_decoder(
            self.hx,
            channel_probs=np.ones(self.n)*(physical_error_rate*2/3),
            max_iter=10,
            bp_method="minimum_sum",
            ms_scaling_factor= 0.625,
            osd_method="osd_cs",
            osd_order=3,
        )

        # decoder for X-errors
        self.bpd_x = bposd_decoder(
            self.hz,
            channel_probs=np.ones(self.n)*(physical_error_rate*2/3),
            max_iter=10,
            bp_method="minimum_sum",
            ms_scaling_factor= 0.625,
            osd_method="osd_cs",
            osd_order=3,
        )
        
    
    def Get_precise_logical_error_rate(self,physical_error_rate=0.0001,trail=1000,block = 14):
        '''using the depolorizing channel to calculate the logical error rate'''
        self.init_decoder(physical_error_rate)
        
        
        # TODO
        p_l = 0
        PL = [0]
        for n_e in range(1,self.n+1):
            p_ne = binomial_probability(self.n,n_e,physical_error_rate)

            if n_e <= block+1 : 
                p_l_ne = self.P_L_given_n_e(n_e,physical_error_rate,trail)
                print('n_e:',n_e)
            else:
                p_l_ne = 1
            p_l += p_ne*p_l_ne
            PL.append(p_l_ne)

        return p_l,PL
    def P_L_given_n_e(self,n_e,physical_error_rate=0.0001,trail=1000):
        '''calculate the probability of logical error given n_e error'''
        # TODO
        p_l_ne = 0
        
        for t in range(trail):
            error_x,error_z = self.Get_error(n_e)
            
            p_l_ne += self.Single_run(error_x,error_z)
        # print(p_l_ne/trail)
        
        return p_l_ne/trail
    def Single_run(self,error_x,error_z):
        # print(f'error_x,error_z:{error_x,error_z}')
        # decode z
        synd_z = self.hx@error_z % 2
        self.bpd_z.decode(synd_z)
        
        #update the channel probability
        # self._channel_update()

        synd_x = self.hz@error_x % 2
        self.bpd_x.decode(synd_x)
        
        

        # print(error_z)
        # print(self.bpd_z.osdw_decoding)
        residual_x = (error_x+self.bpd_x.osdw_decoding) % 2
        residual_z = (error_z+self.bpd_z.osdw_decoding) % 2
        # print(f'residual_x,residual_z:{residual_z}')
        if (self.lx@residual_x % 2==0).all() and (self.lz@residual_z % 2 == 0).all():
            return 0
        else:
            return 1
    def Get_error(self,d_e):
        '''sample an error of weight d_e under the depolorizing model'''
        error_x = np.zeros(self.n).astype(int)
        error_z = np.zeros(self.n).astype(int)
        
        indices = np.random.choice(self.n, d_e, replace=False)
        for index in indices:
            error_type = np.random.choice(['x', 'z','y'])
            if error_type == 'x':
                error_x[ index] = 1
            elif error_type == 'z':
                error_z[ index] = 1
            else:
                error_x[ index] = 1
                error_z[ index] = 1
        return error_x, error_z
    def _channel_update(self,update_direction):

        '''
        Function updates the channel probability vector for the second decoding component
        based on the first. The channel probability updates can be derived from Bayes' rule.
        '''

        #x component first, then z component
        if update_direction=="x->z":
            decoder_probs=np.zeros(self.N)
            for i in range(self.N):
                if self.bpd_x.osdw_decoding[i]==1:
                    if (self.channel_probs_x[i]+self.channel_probs_y[i])==0:
                        decoder_probs[i]=0
                    else:
                        decoder_probs[i]=self.channel_probs_y[i]/(self.channel_probs_x[i]+self.channel_probs_y[i])
                elif self.bpd_x.osdw_decoding[i]==0:
                        decoder_probs[i]=self.channel_probs_z[i]/(1-self.channel_probs_x[i]-self.channel_probs_y[i])
        
            self.bpd_z.update_channel_probs(decoder_probs)

        #z component first, then x component
        elif update_direction=="z->x":
            self.bpd_z.osdw_decoding
            decoder_probs=np.zeros(self.N)
            for i in range(self.N):
                if self.bpd_z.osdw_decoding[i]==1:
                    
                    if (self.channel_probs_z[i]+self.channel_probs_y[i])==0:
                        decoder_probs[i]=0
                    else:
                        decoder_probs[i]=self.channel_probs_y[i]/(self.channel_probs_z[i]+self.channel_probs_y[i])
                elif self.bpd_z.osdw_decoding[i]==0:
                        decoder_probs[i]=self.channel_probs_x[i]/(1-self.channel_probs_z[i]-self.channel_probs_y[i])

            
            self.bpd_x.update_channel_probs(decoder_probs)


    

if __name__ == '__main__':
    # h = np.array([[1,1,0,0,0,0,1],[1,0,0,1,1,0,1],[0,1,1,0,1,1,0],[0,0,0,1,1,1,1]])
    from parameter_converter import Parameters_converter
    converter = Parameters_converter(method = 'HGP',hyperparameters={'p': 4, 'q': 4, 'm': 3})
    from Construct_Code import CSS
    from Evaluate import CSS_Evaluator
    code1 = np.array([1, 3, 0, 3, 2, 0, 1, 2, 2, 1, 3, 0, 0, 1, 0, 3])
    code1 = converter.generate_parameters(code1)
    css_instance_1 = CSS()
    css_instance_1.construct(method='HGP',para_dict=code1)

    import pyldpc

    h, _ = pyldpc.make_ldpc(6, 2,3)
    # print(h)
    h =np.array ([[ 1,1,1,1,1,1,0,0,0,0,0,0],[ 0,0,0,0,0,0,1,1,1,1,1,1]])
    surface_code = hgp(h,h)
    Hx = surface_code.hx
    Hz = surface_code.hz
    Hx = np.array([[1,1,1,0,0,0,1,0],[0,0,1,0,1,1,1,0],[1,1,0,1,0,0,0,1],[0,0,0,1,1,1,0,1]])
    Hz = np.array([[0,1,0,0,1,0,1,1],[0,1,1,1,1,0,0,0],[1,0,0,0,0,1,1,1],[1,0,1,1,0,1,0,0]])
    Hx = css_instance_1.HX
    Hz = css_instance_1.HZ

    CSS_evaluator = CSS_Evaluator(Hx,Hz)
    returns = CSS_evaluator.Get_error_rate(physical_error_rate=0.02)
    return2 = CSS_evaluator.Get_precise_logical_error_rate(physical_error_rate=0.02,trail=300,block=10)
    print(returns)
    print(return2)
    # print(type(returns))

    # plt.plot(PL, label='simple Monte Carlo', color='g')

    # plt.xlabel('n_e')
    # plt.ylabel('p(logical error|n_e)')
    # plt.title('p(logical error|n_e) vs n_e')
    # # plt.legend()
    # plt.grid(True)


    PP = np.linspace(0.5, 7, 15)
    PP = 10**(-PP)   
    print(f'PP:{PP}')
    p_l_acc = np.array([CSS_evaluator.Get_precise_logical_error_rate(physical_error_rate=p,trail=1000,block=10) for p in PP])
    p_l = np.array([CSS_evaluator.Get_error_rate(physical_error_rate=p) for p in PP])
    print(f'PP:{PP}')
    print(f'p_l_acc:{p_l_acc}')
    print(f'p_l:{p_l}')
    plt.xscale('log')

    plt.yscale('log')
    plt.plot(PP, p_l, label='simple Monte Carlo', color='g')
    plt.plot(PP, p_l_acc, label='Precise', color='r')
    plt.xlabel('PP')
    plt.ylabel('P_l')
    plt.title('Two estimates of P_l')
    plt.legend()
    plt.grid(True)
    plt.show()