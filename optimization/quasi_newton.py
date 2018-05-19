import numpy as np
import optimization.multivariable_calculus as mvc

def best(f, criticalPoints):
    globalOptimum = criticalPoints[0]
    optimumOutput = f(globalOptimum)
    for i in range(1, len(criticalPoints)):
        contender = f(criticalPoints[i])
        if contender < optimumOutput:
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
    alpha = 1e2
    control = 0.5 # 0 < control < 1 is a control parameter for the Armijo-Goldstein condition. See https://en.wikipedia.org/wiki/Backtracking_line_search.
    lesseningFactor = 0.5 # 0 < lesseningFactor < 1 is multiplied into alpha at each iteration to lessen it.
    m = pk.dot(grad.T).tolist()[0] # local slope in direction pk
    t = control * m # Store this value for later access in the condition.
    fxk = f(xk.tolist()[0])
    # Armijo set control and lesseningFactor to 1/2 in his original paper, as done here.
    # Now, lessen alpha until the condition is satisfied. Break after 40 steps in case something went wrong.
    for i in range(40):
        # If the Armijo-Goldstein condition is met, terminate. Otherwise, lessen alpha.
        if f((xk[0] + alpha*pk).tolist()) <= fxk + alpha*t:
            break
        alpha = alpha * lesseningFactor
    return alpha

def BFGS(V, sk, yk):
    ykskT = yk.dot(sk.T)
    return (1 + yk.dot(V).dot(yk.T)/ykskT)*(sk.T).dot(sk)/ykskT - (V.dot(yk.T).dot(sk) + (sk.T).dot(yk).dot(V))/ykskT #(1 + (yk.T).dot(V).dot(yk)/skTyk)*(sk.dot(sk.T))/skTyk - (sk.dot(yk.T) * V+ V.dot(yk).dot(sk.T))/skTyk

def DFP(V, sk, yk):
    VykT = V.dot(yk.T)
    return (sk.T).dot(sk)/yk.dot(sk.T) - (VykT.dot(yk).dot(V))/yk.dot(VykT)

def optimize(f, n, convergence = 1e-6, trials=1, maxSteps = 100, lowerBound = -1, upperBound = 1):
    criticalPoints = []
    convergenceSquared = convergence**2
    for trial in range(trials):
        # Initial guess for optimum, to be optimized
        xk = np.array([[np.random.random() * (upperBound-lowerBound) + lowerBound for i in range(n)]])#np.array([0, -0.1])
        # This runs a combination of the DFP and BFGS Quasi-Newton methods for optimization.
        V = np.array(identityMatrix(n)) # Initially, the inverse Hessian is approximated with the identity matrix.
        for i in range(maxSteps):
            grad = np.array([mvc.gradient(f, xk.tolist()[0])])
            # Calculate direction by Newton's Method with an approximated inverse Hessian.
            pk = -V.dot(grad[0])
            # Perform line search to calculate step size, alpha.
            alpha = lineSearch(f, n, grad, xk, pk) # Calculate to find next point after step.
            sk = alpha*pk # step at iteration k
            # If the step size is small enough, terminate search. Further computation would be wasteful.
            if sk.dot(sk) < convergenceSquared:
                continue
            sk = np.array([sk.tolist()])
            xk_next = xk + sk
            # Now, for the new xk, update the approximate inverse Hessian with BFGS.
            yk = np.array([mvc.gradient(f, xk_next.tolist()[0])]) - grad
            xk = xk_next
            #### Update the inverse Hessian with a combination of DFP and BFGS. For more details, see en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm.
            ####t = (2*alpha - 1)/alpha
            ####V = V + t * DFP(V, sk, yk) + (1-t) * BFGS(V, sk, yk)
            V = V + BFGS(V, sk, yk)
        criticalPoints.append(xk.tolist()[0])
    return best(f, criticalPoints)
