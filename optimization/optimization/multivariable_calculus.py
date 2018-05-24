import numpy as np

def mixedPartialDerivative(f, v, order):
    # order is an array of dimensions by which to take partial derivatives.
    # order = [0, 1]    ==> partial of f with respect to x then y.
    # order = [1, 2, 2] ==> partial of f with respect to y then z then z.
    h = 1e-6 # The limiting variable in the limit definition of the partial derivative.
    if len(order) == 1:
        # If order is 1, numerically approximate the partial derivative as usual.
        step = v[:]
        step[order[0]] += h
        return (f(step) - f(v)) / h
    # Peel the onion of layers: first, the outermost, then work inwards.
    step = v[:]
    step[order[len(order) - 1]] += h # Final derivative step done first.
    # Recursively calculate partial derivatives for the final derivative.
    return (mixedPartialDerivative(f, step, order[0:len(order) - 1]) - mixedPartialDerivative(f, v, order[0:len(order) - 1])) / h

def hessian(f, v):
    H = []
    for i in range(len(v)):
        H.append([])
        for j in range(len(v)):
            H[i].append(mixedPartialDerivative(f, v, [i, j]))
    return H

# gradient computes the gradient of a function of n dimensions, f, at position v.
def gradient(f, v):
    grad = [0 for i in range(len(v))] # This is the gradient array of partial derivatives.
    h = 1e-8 # The limiting variable in the limit definition of the partial derivative is approximated with 1e-8.
    for i in range(len(v)):
        step = v[:]
        step[i] += h
        grad[i] = (f(step) - f(v)) / h # limit definition of the partial derivative
    return grad
