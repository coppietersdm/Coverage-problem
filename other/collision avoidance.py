import numpy as np
import matplotlib.pyplot as plt

d_crit = 0.2
R = 1

v_crit = 2*R - np.sqrt(R*R - d_crit*d_crit)
print(v_crit)

def f(d):
    e = d/(2*d_crit)
    return 4*v_crit*d_crit*(e-1)**2 * (e<1)*(e!=0)

def fd(d):
    e = d/(2*d_crit)
    return 4*v_crit*d_crit*2*(e-1)/(2*d_crit) * (e<1)*(e!=0)

def fdd(d):
    e = d/(2*d_crit)
    return 4*v_crit*d_crit*2/(2*d_crit) * (e<1)*(e!=0)


def avoidance(x):
    x = x.reshape(-1,2)
    return np.sum([[f(np.linalg.norm(i-j)) for i in x] for j in x])


def avoidance_d(x):
    x = x.reshape(-1,2)
    gradient = x*0
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            if(i!=j):
                dij = np.linalg.norm(xi-xj)
                gradient[i] += fd(dij)*(xi-xj)/dij
                gradient[j] += fd(dij)*(xj-xi)/dij
            
    return gradient.reshape(-1)

def avoidance_dd(x):
    hessian = np.zeros(len(x), len(x))
    x = x.reshape(-1,2)
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            if(i<j):
                subhessian = hessian*0
                dij = np.linalg.norm(xi-xj)
                grad_i_dij = fd(dij)*(xi-xj)/dij
                h = (dij*np.eye(2) - np.outer(grad_i_dij,(xi-xj)))/(dij*dij)
                subhessian[2*i:2*i+2,2*j:2*j+2] = h
                subhessian[2*j:2*j+2,2*i:2*i+2] = -h
                subhessian[2*i:2*i+2,2*j:2*j+2] = -h
                subhessian[2*j:2*j+2,2*j:2*j+2] = h
    hessian/2
                
                
    
    

x = np.random.rand(10)
dx = np.random.rand(10)/100
print(avoidance(x+dx) - avoidance(x), dx@avoidance_d(x))
