import numpy as np

def f1(v):
    return 100 * (v[1] - v[0]*v[0])**2 + (1 - v[0])**2

def f2(v): # Negative 3d analogy of bell curve
    return -np.exp(-(v[0]*v[0] + v[1]*v[1]))
# -2*e^(-(x^2 + y^2)) + 4*x*x*e^(-(x^2 + y^2))

def f3(v):
    return v[0]*v[0]*v[0] + v[0]*v[0] + v[0]*v[1]

def f4(v):
    return (v[0] - 2)*(v[0] - 2) + v[1]*v[1]
