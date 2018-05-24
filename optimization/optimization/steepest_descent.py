import numpy as np
import optimization.multivariable_calculus as mvc

def best(f, criticalPoints, maximize):
    globalOptimum = criticalPoints[0]
    optimumOutput = f(globalOptimum)
    for i in range(1, len(criticalPoints)):
        contender = f(criticalPoints[i])
        if (not maximize and contender < optimumOutput) or (maximize and contender > optimumOutput):
            globalOptimum = criticalPoints[i]
            optimumOutput = contender
    return globalOptimum

# optimize uses gradient descent to minimize or maximize arbitrary multivariable functions. optimize should only take functions that don't explode.
# This means the function shouldn't have points at negative infinity if it's being minimized, for example.
def optimize(f, n, convergence = 1e-4, trials = 4, maxSteps = 1000, stepSize = 0.01, lowerBound = -1, upperBound = 1, maximize = False):
    '''
    optimize takes a function (f), its dimension (n), the range in which it should initialize coordinates for optimization,
    how many trials to run in search for a global minimum (trials), how big steps should be (stepSize),
    how many times to run gradient descent (steps), whether or not to normalize the gradient before adding it (normalize),
    and a boolean on whether to maximize or minimize (minimize). It applies gradient descent/ascent steps times to locate a critical point.
    It repeats this process trials times and returns the lowest/highest of the local minima/maxima.
    Don't use rapidly changing functions with optimize. The higher gradients get, the lower stepSize must be set to avoid explosion.
     '''
    criticalPoints = [] # Stores local critical points as 1x2 arrays: [value, position].
    for trial in range(trials):
        # Initialize random coordinates within bound of the origin.
        coordinates = np.array([np.random.random() * (upperBound - lowerBound) + lowerBound for i in range(n)])
        for step in range(maxSteps):
            # Run steps iterations of gradient descent on the random coordinates.
            grad = np.array(mvc.gradient(f, coordinates.tolist()))
            magnitude = np.sqrt(grad.dot(grad))
            if magnitude < convergence:
                break # If the gradient is sufficiently small, there's no point pursuing further gains.
            grad = grad / magnitude # Normalize the gradient.
            for i in range(len(coordinates)):
                coordinates[i] += (stepSize if maximize else -stepSize) * grad[i]
        criticalPoints.append(coordinates.tolist())
    return best(f, criticalPoints, maximize)
