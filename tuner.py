import numpy as np
import optimization.multivariable_calculus as mvc
import optimization.steepest_descent as sd
import optimization.newton as newton
import optimization.quasi_newton as qn
import RNN as rnn

def networkLoss(v):
  averageLoss = 0
  for i in range(1): # v[0] and v[1] must be whole numbers
    hypotheticalRNN = rnn.RNN(v[0], v[1], v[2])
    hypotheticalRNN.initAdagrad()
    averageLoss += hypotheticalRNN.train(100)#0000)#/v[1]
  return averageLoss

def continuousNetworkLoss(v):
  # print(v)
  # To make the discrete inputs of hidden_size and seq_length continuous for the loss function,
  # take their weighted averages between the two closest integers.
  lowHiddenWeight  = 1 - (v[0] - np.floor(v[0]))
  highHiddenWeight = v[0] - np.floor(v[0])
  lowSeqWeight  = 1 - (v[1] - np.floor(v[1]))
  highSeqWeight = v[1] - np.floor(v[1])
  discreteLowHidden  = lowHiddenWeight * (lowSeqWeight*networkLoss([int(np.floor(v[0])), int(np.floor(v[1])), v[2]]) +
                                         highSeqWeight*networkLoss([int(np.floor(v[0])), int(np.floor(v[1])) + 1, v[2]]))
  discreteHighHidden = highHiddenWeight * (lowSeqWeight*networkLoss([int(np.floor(v[0])) + 1, int(np.floor(v[1])), v[2]]) +
                                          highSeqWeight*networkLoss([int(np.floor(v[0])) + 1, int(np.floor(v[1])) + 1, v[2]]))
  return discreteLowHidden + discreteHighHidden

class HPT: # HyperparameterTuner

  def __init__(self, network, tuning_rate = 1e-2, training_steps = 1e3, optimizer=sd.optimize):
    '''
    Hyperparameter tuners find the best combination of hyperparameters to make RNNs learn the fastest.
    They pick a random combination of hyperparameters, first, and perform gradient descent to minimize loss after a constant number of training steps.
    The scalar multiplied into the gradient at each step is tuning_rate.
    The constant number of training steps used on networks to evaluate combinations is training_steps.
    '''
    self.network = network
    self.tuning_rate = tuning_rate
    self.training_steps = training_steps
    self.optimizer = optimizer

  def tune(self):
    # apply Gradient Descent on randomly initialized network hyperparameters.
    # note: RNN.train returns the loss of the network following the training.
    return self.optimizer(continuousNetworkLoss)

character_model = rnn.RNN()
# character_model.train(1000)
# character_model.sampleN(200)

# Steepest Descent Tuners #[CHANGE STEPS BACK TO EXPECTED]
sdTuner10 = HPT(character_model, optimizer = lambda net: sd.optimize
                         (net, 3, maxSteps=10, stepSize=0.1, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
sdTuner50 = HPT(character_model, optimizer = lambda net: sd.optimize
                         (net, 3, maxSteps=50, stepSize=0.1, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
sdTuner100 = HPT(character_model, optimizer = lambda net: sd.optimize
                         (net, 3, maxSteps=100, stepSize=0.1, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
sdTuner500 = HPT(character_model, optimizer = lambda net: sd.optimize
                         (net, 3, maxSteps=500, stepSize=0.1, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
sdTuner1000 = HPT(character_model, optimizer = lambda net: sd.optimize
                         (net, 3, maxSteps=1000, stepSize=0.1, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
sdTuner1000AntiExt = HPT(character_model, optimizer = lambda net: sd.optimize
                         (net, 3, maxSteps=1000, stepSize=0.01, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
sdTuner1000Ext = HPT(character_model, optimizer = lambda net: sd.optimize
                         (net, 3, maxSteps=1000, stepSize=0.5, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
sdTuner5000 = HPT(character_model, optimizer = lambda net: sd.optimize
                         (net, 3, maxSteps=5000, stepSize=0.1, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
sdTuner10000 = HPT(character_model, optimizer = lambda net: sd.optimize
                         (net, 3, maxSteps=10000, stepSize=0.1, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
sdTuner10000AntiExt = HPT(character_model, optimizer = lambda net: sd.optimize
                         (net, 3, maxSteps=10000, stepSize=0.01, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
sdTuner10000Ext = HPT(character_model, optimizer = lambda net: sd.optimize
                         (net, 3, maxSteps=10000, stepSize=0.5, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))

# Newton's Method Tuners #[CHANGE STEPS BACK TO EXPECTED]
#newtonTuner10 = HPT(character_model, optimizer = lambda net: newton.optimize
#                         (net, 3, maxSteps=10, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
#newtonTuner100 = HPT(character_model, optimizer = lambda net: newton.optimize
#                         (net, 3, maxSteps=100, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
#newtonTuner1000 = HPT(character_model, optimizer = lambda net: newton.optimize
#                         (net, 3, maxSteps=1000, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))

# Quasi-Newton Tuners #[CHANGE STEPS BACK TO EXPECTED]
#qnTuner10 = HPT(character_model, optimizer = lambda net: qn.optimize
#                         (net, 3, maxSteps=10, lowerBound = 1, upperBound = 10, convergence=1e-4))
#qnTuner100 = HPT(character_model, optimizer = lambda net: qn.optimize
#                         (net, 3, maxSteps=100, lowerBound = 1, upperBound = 10, convergence=1e-4))
#qnTuner1000 = HPT(character_model, optimizer = lambda net: qn.optimize
#                         (net, 3, maxSteps=1000, lowerBound = 1, upperBound = 10, convergence=1e-4))

def write():
  with open("output.txt", "w") as text_file:
        text_file.write(output)

output = "HYPERPARAMETER TUNING: (Objective function is 10000 training steps averaged 10 times. 1000 training steps averaged 10 times takes about 400 seconds.)\n\n"
output += "STEEPEST DESCENT:\n"
output += "10 steps (stepSize = 0.1): %s\n" %(sdTuner10.tune())
write()
print("10 down")
output += "50 steps (stepSize = 0.1): %s\n" %(sdTuner50.tune())
write()
print("50 down")
output += "100 steps (stepSize = 0.1): %s\n" %(sdTuner100.tune())
write()
print("100 down")
output += "500 steps (stepSize = 0.1): %s\n" %(sdTuner500.tune())
write()
print("500 down")
output += "1000 steps (stepSize = 0.1): %s\n" %(sdTuner1000.tune())
write()
print("1000 down")
output += "1000 steps (stepSize = 0.01): %s\n" %(sdTuner1000AntiExt.tune())
write()
print("1000 (anti-ext) down")
output += "1000 steps (stepSize = 0.5): %s\n" %(sdTuner1000Ext.tune())
write()
print("1000 (ext) down")
output += "5000 steps (stepSize = 0.1): %s\n" %(sdTuner5000.tune())
write()
print("5000 down")
output += "10000 steps (stepSize = 0.1): %s\n" %(sdTuner10000.tune())
write()
print("10000 down")
output += "10000 steps (stepSize = 0.01): %s\n" %(sdTuner10000AntiExt.tune())
write()
print("10000 (anti-ext) down")
output += "10000 steps (stepSize = 0.5): %s\n" %(sdTuner10000Ext.tune())
write()
print("10000 (ext) down")
#output += "\nNEWTON'S METHOD:\n10 steps: %s\n100 steps: %s\n1000 steps: %s\n" %(newtonTuner10.tune(), newtonTuner100.tune(), newtonTuner1000.tune())
#output += "\nQUASI-NEWTON:\n10 steps: %s\n100 steps: %s\n1000 steps: %s\n" %(qnTuner10.tune(), qnTuner100.tune(), qnTuner1000.tune())
