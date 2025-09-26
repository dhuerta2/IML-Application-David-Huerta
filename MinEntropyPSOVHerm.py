# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 19:22:00 2025

@author: sable
"""

import numpy as np

def entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    return(-np.sum(eigvals*np.log(eigvals)))
p = .1
def deph(rho):
    rho_QC = (1-p)*rho + 0.5*p*np.identity(2)
    return rho_QC
#bounds = [-1,1]
def hermitize(rho):
    rho = rho.conj().T@rho
    rho = rho/np.trace(rho)
    return rho

def particle_swarm(
        func=entropy,
        dim = 2,
        num_particles = 100,
        max_iter = 200,
        w = 0.7,
        beta = 1.5,
        gamma = 2):
    

    
    particles = np.random.rand(num_particles, dim, dim)
    v = np.random.rand(num_particles, dim, dim)*0.25
    
    personal_best_loc = np.copy(particles)#baseline
    global_best_loc = 0
    global_best_obj = np.Infinity

    itera = 0
    while itera < max_iter:
        for i in range(0,len(particles)):#each particle moves 1 step
            
            if itera ==0:
                rho = hermitize(particles[i])
                particles[i] = rho

            r1 = np.random.uniform(0,1)
            r2 = np.random.uniform(0,1)
            v[i] = w*v[i] + beta*r1*(personal_best_loc[i] - particles[i]) + gamma*r2*(global_best_loc -particles[i])
            v[i] = hermitize(v[i])#dunno if this is necessary
                                  #seems to perform better
                                  #then without hermitizing v
            particles[i] = particles[i] + v[i]
            rho = hermitize(particles[i])
            particles[i] = rho
 
            cur_obj = func(deph(rho))
            if cur_obj < global_best_obj:
                global_best_obj = cur_obj
                global_best_loc = rho
                eig_val = np.linalg.eigvalsh(rho)
            if cur_obj < func(personal_best_loc[i]):
                personal_best_loc[i] = rho

        itera +=1
        if itera % 20 == 0:
            print(f"Iter {itera}: Best Score = {func(deph(global_best_loc))}")
    return(global_best_loc, global_best_obj, eig_val)

print(particle_swarm())
print("\n")
correct = -(p/2)*np.log(p/2)-(1-(p/2))*np.log((1-(p/2)))
print(correct)