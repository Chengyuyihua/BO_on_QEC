"""
This file contains the code for the evaluation of the stabilizer codes using
"""
from typing import Tuple
from code_construction.code_construction import StabilizerCode
import numpy as np
class Undetectable_error_rate():
    def __init__(self,stabilizer_code:StabilizerCode,noise_model=(1,1,1)):

        self.stabilizer_code = stabilizer_code
        self.noise_model = noise_model
    def evaluate(self,p:float)->float:
        """
            return the undetectable error rate of the stabilizer code (n<20, otherwise the run time is massive)
        """
        
        H = self.stabilizer_code.h
        r, m = H.shape
        n = m//2

        S_set = set()
        for i in range(2 ** r):

            coeffs = np.array(list(map(int, np.binary_repr(i, width=r))))

            v = np.mod(coeffs.dot(H), 2)
            S_set.add(tuple(v.tolist()))
        

        undetectable_rate = 0.0
        for i in range(4 ** n):

            e = np.array(list(map(int, np.binary_repr(i, width=n*2))))

            syndrome = np.mod(H.dot(e), 2)
            if np.all(syndrome == 0): 
                if tuple(e.tolist()) not in S_set: 
                    prob = self.p(p,e)
                    undetectable_rate += prob
        
        return undetectable_rate
    def p(self,p,e):
        """
            return the probability of the error e
        """
        n = len(e)//2
        p_x,p_y,p_z =  self.noise_model
        p_x = p_x*p
        p_y = p_y*p
        p_z = p_z*p
        total_prob = 1.0
        ex = e[:n]
        ez = e[n:]

        for i in range(n):

            if ex[i] == 0 and ez[i] == 0:
                total_prob *= (1 - p)
            elif ex[i] == 1 and ez[i] == 0:

                total_prob *= p_x
            elif ex[i] == 0 and ez[i] == 1:

                total_prob *= p_z
            elif ex[i] == 1 and ez[i] == 1:
                total_prob *= p_y
        return total_prob
        


if __name__ == '__main__':
    from code_construction.code_construction import CodeConstructor
    constructor = CodeConstructor(method='canonical',para_dict={'n':5,'k':1,'r':2})
    stabilizer_code = constructor.construct()
    undetectable_error_rate = Undetectable_error_rate(stabilizer_code)
    print(undetectable_error_rate.evaluate())