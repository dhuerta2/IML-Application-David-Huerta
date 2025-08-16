# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 10:53:11 2025

@author: dhuerta
"""

#Pattern Search
import numpy as np

def shubert(x):
    x = np.asarray(x)
    result = 1
    for xi in x:
        inner_sum = np.sum([j * np.cos((j + 1) * xi + j) for j in range(1, 6)])
        result *= inner_sum
    return result

def pattern_search_simple(func,##not very accurate
                   bounds,
                   mesh_const,
                   dir_subset,
                   dim=2):
    
    x_0 = np.random.uniform(bounds[0],bounds[1],dim)
    cur_pos = x_0
    cur_obj = func(cur_pos)   
    mesh = mesh_const
    
    holder_pos = x_0 #allows for cycling through best of all mesh points 
    #without starting a new cycle once a single better objective is found
    while mesh > 0.0000000005:
        
        for velocity in dir_subset:
            mesh_point = cur_pos + velocity*mesh
            for i in range(0,dim):
                if mesh_point[i] < bounds[0]:#lb
                    mesh_point[i] = bounds[0]
                if mesh_point[i] > bounds[1]:#ub
                    mesh_point[i] = bounds[1]
            mobj_val = func(mesh_point)
            if mobj_val < func(holder_pos):
                holder_pos = mesh_point
            else:
                continue
        
        if func(holder_pos) < func(cur_pos):
            mesh *= 2
            cur_pos = holder_pos
        else:
            mesh *=0.5
        cur_obj = func(cur_pos)

    return (cur_pos,cur_obj)
S2D = [np.array([1,0]),np.array([0,1]),np.array([-1,0]),np.array([0,-1])]
S2D_4 = [np.array([1,0]),np.array([0,1]),np.array([-1,0]),np.array([0,-1]), np.array([1,1]) ,np.array([-1,1]),np.array([1,-1]),np.array([-1,-1])]

S2D_12 = np.copy(S2D_4)#includes not just corner vectors but, the vectors between each corner and x/y axes
for i in range(4,8):
    first_arr = np.array([[2,0],[0,1]]@S2D_4[i])
    second_arr = np.array([[1,0],[0,2]]@S2D_4[i])
    S2D_12 = np.vstack([S2D_12, first_arr])
    S2D_12 = np.vstack([S2D_12, second_arr])

#print(S2D_12)
#print(pattern_search_simple(shubert, np.array([-10,10]), 4, S2D_4))
def testing(num_attempts, dir_array):
    num = 0
    correct = 0
    for attempts in range(0,num_attempts):
        value = pattern_search_simple(shubert, np.array([-10,10]), 4, dir_array)[1]
        if round(value,4) == -186.7309:
            correct += 1
        num +=1
    return (correct/num)
print(testing(200,S2D_4))