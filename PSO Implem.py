# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 08:10:23 2025

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

def cross_in_tray(x):
    x = np.asarray(x)
    x1 = x[0]
    x2 = x[1]
    
    frac = np.sqrt(x1**2 + x2**2)/np.pi
    
    result = -0.0001*(np.abs( np.sin(x1) * np.sin(x2) * np.exp(np.abs(100-frac))) + 1)**(0.1)
    return result

def particle_swarm(
        func,
        bounds,
        dim = 2,
        num_particles = 50,
        max_iter = 200,
        w = 0.7,
        beta = 1.5,
        gamma = 1.5):
    

    
    particles = np.random.uniform(bounds[0],bounds[1],(num_particles,dim))
    v = np.random.uniform(bounds[0],bounds[1],(num_particles,dim))*2#velocity

    personal_best_loc = np.copy(particles)#baseline
    global_best_loc = 0
    global_best_obj = np.Infinity

    itera = 0
    while itera < max_iter:
        for i in range(0,len(particles)):#each particle moves 1 step
            r1 = np.random.uniform(0,1)
            r2 = np.random.uniform(0,1)
            v[i] = w*v[i] + beta*r1*(personal_best_loc[i] - particles[i]) + gamma*r2*(global_best_loc -particles[i])
            particles[i] = particles[i] + v[i]
            for j in range(0,dim):
                if particles[i][j] < bounds[0]:#lb
                    particles[i][j] = bounds[0]
                if particles[i][j] > bounds[1]:#ub
                    particles[i][j] = bounds[1]
            
            cur_obj = func(particles[i])
            if cur_obj < global_best_obj:
                global_best_obj = cur_obj
                global_best_loc = particles[i]
            if cur_obj < func(personal_best_loc[i]):
                personal_best_loc[i] = particles[i]

        itera +=1
        if itera % 20 == 0:
            print(f"Iter {itera}: Best Score = {func(global_best_loc)}")
    return(global_best_loc, global_best_obj)


#print(particle_swarm(shubert, np.array([-10,10])))

print(particle_swarm(eggholder,np.array([-512,512])))

#print(particle_swarm(cross_in_tray,np.array([-10,10])))