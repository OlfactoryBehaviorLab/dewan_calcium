# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:22:49 2022

@author: VincisLabMain
"""

def Sliding_probability(cur_bin, maxval, minval, increments):
    import numpy as np
   
    Prob_vector = list()
    #st_range = 0
    mid_range = (maxval-minval)/increments
    end_range = maxval
    critvalues = np.arange(minval,end_range,mid_range)
    # if maxval==0:
    #     Prob_vector = [0]*len(critvalues)
    # else:
    for i in range(len(critvalues)):
        Prob_vector.append(sum(np.ravel(cur_bin > critvalues[i]))/len(np.ravel(cur_bin)))
           
    return Prob_vector