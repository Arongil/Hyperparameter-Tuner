import numpy as np
import multivariable_calculus as mvc

###
import example_functions as fn
###

def best(f, criticalPoints, maximize):
    globalOptimum = criticalPoints[0]
    optimumOutput = f(globalOptimum)
    for i in range(1, len(criticalPoints)):
        contender = f(criticalPoints[i])
        if (not maximize and contender < optimumOutput) or (maximize and contender > optimumOutput):
            globalOptimum = criticalPoints[i]
            optimumOutput = contender
    return globalOptimum

def identityMatrix(n):
    I = []
    for i in range(n):
        I.append([])
        for j in range(n):
            I[i].append(1 if i == j else 0)
    return I

def lineSearch(f, n, grad, xk, pk):
    # Given a function and a in which direction to travel,
    # lineSearch solves for the optimal distance to travel to not under- or over-shoot.
    # Backtracking line search will initialize alpha, the distance to travel, as a high number.
    # alpha will be iteratively lessened until the Armijo-Goldstein condition is satisfied.
    alpha = 1e1
    control = 0.5 # 0 < control < 1 is a control parameter for the Armijo-Goldstein condition. See https://en.wikipedia.org/wiki/Backtracking_line_search.
    lesseningFactor = 0.5 # 0 < lesseningFactor < 1 is multiplied into alpha at each iteration to lessen it.
    m = (pk.T).dot(grad) # local slope in direction pk
    t = -control * m # Store this value for later access in the condition.
    # Armijo set control and lesseningFactor to 1/2 in his original paper, as done here.
    # Now, lessen until the condition is satisfied. Break after 100 steps if something went wrong.
    for i in range(100):
        # If the Armijo-Goldstein condition is met, terminate. Otherwise, lessen alpha.
        if f(xk.tolist()) - f((xk + alpha*pk).tolist()) >= alpha*t:
            break
        alpha = alpha * lesseningFactor
    print("%s: %s" %(i, alpha))
    return alpha

def Broyden(V, sk, yk):
    Vyk = V.dot(yk)
    return V + (sk - Vyk).dot((sk.T).dot(V))/((sk.T).dot(Vyk))

def BFGS(V, sk, yk):
    skTyk = (sk.T).dot(yk)
    return V + ((skTyk + (yk.T).dot(V).dot(yk)) * sk.dot(sk.T))/(skTyk**2) - (V.dot(yk).dot(sk.T) + sk.dot(yk.T) * V)/skTyk

def DFP(V, sk, yk):
    Vyk = V.dot(yk)
    return V + (sk.dot(sk.T))/((sk.T).dot(yk)) - (Vyk.dot(yk.T) * V)/((yk.T).dot(Vyk))

def optimize(f, n, convergence = 1e-4, trials=1, maxSteps = 10, lowerBound = -1, upperBound = 1, maximize=False):
    criticalPoints = []
    convergenceSquared = convergence**2
    for trial in range(trials):
        # Initial guess for optimum, to be optimized
        xk = np.array([-1.2, 1])#np.array([np.random.random() * (upperBound-lowerBound) + lowerBound for i in range(n)])
        # print(xk)
        # This runs the BFGS Quasi-Newton method for optimization.
        V = np.array(identityMatrix(n)) # Initially, the inverse Hessian is approximated with the identity matrix.
        for i in range(maxSteps):
            # print(V)
            grad = np.array(mvc.gradient(f, xk.tolist()))
            if grad.dot(grad) < convergenceSquared: # x dot x is the squared magnitude of x
                break # Optimum reached.
            # Calculate direction by Newton's Method with an approximated inverse Hessian.
            pk = -V.dot(grad)
            pk = pk / np.sqrt(pk.dot(pk)) # Normalize pk.
            # Perform line search to calculate step size alpha.
            alpha = lineSearch(f, n, grad, xk, pk)
            # Calculate to find next point after step.
            sk = alpha*pk # Step at iteration k
            xk_next = xk + sk
            # Now, for the new xk, update the approximate inverse Hessian with BFGS.
            yk = np.array(mvc.gradient(f, xk_next.tolist())) - np.array(mvc.gradient(f, xk.tolist()))
            # print(yk)
            xk = xk_next
            # Update the inverse Hessian with BFGS. For more details, see en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm.
            V = BFGS(V, sk, yk)
        criticalPoints.append(xk.tolist())
    return best(f, criticalPoints, maximize)
