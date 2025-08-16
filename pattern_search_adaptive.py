# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 14:31:10 2025

@author: dhuerta
"""
import numpy as np

def shubert(x):
    x = np.asarray(x)
    result = 1
    for xi in x:
        inner_sum = np.sum([j * np.cos((j + 1) * xi + j) for j in range(1, 6)])
        result *= inner_sum
    return result
def eggholder(x):
    x = np.asarray(x)
    x1 = x[0]
    x2 = x[1]
    result = -(x2+47)*np.sin(np.sqrt(np.abs(x2+(x1/2)+47)))-x1*np.sin(np.sqrt(np.abs(x1-(x2+47))))
    return result

def pattern_search_adaptive(func,
                       bounds,
                       mesh_const,
                       dim=2):
        
        x_0 = np.random.uniform(bounds[0],bounds[1],dim)
        cur_pos = x_0
        cur_obj = func(cur_pos)   
        mesh = mesh_const
        
        #holder allows for cycling through best of all mesh points 
        #without starting a new cycle once a single better objective is found
        while mesh > 0.0000000005:
            holder_pos = cur_pos
            v_pos = np.random.uniform(bounds[0],bounds[1],(10 ,dim))
            v_neg = -v_pos
            v_orth = np.multiply(v_pos,v_neg)#orthogonal to our random gen.
            velocities = np.vstack([v_pos, v_neg])
            velocities = np.vstack([velocities,v_orth])
            
            for velocity in velocities:
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
            continue;
        return (cur_pos, cur_obj)

print(pattern_search_adaptive(shubert,np.array([-10,10]), 4))

#print(pattern_search_adaptive(eggholder,np.array([-512,512]), 4))

def testing_shub(num_attempts):
    num = 0
    correct = 0
    for attempts in range(0,num_attempts):
        value = pattern_search_adaptive(shubert, np.array([-10,10]), 4)[1]
        if round(value,4) == -186.7309:
            correct += 1
        num +=1
    return (correct/num)
#print(testing_shub(200))