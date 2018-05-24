import numpy as np
import optimization.multivariable_calculus as mvc
import optimization.steepest_descent as sd
import optimization.newton as newton
import optimization.quasi_newton as qn
import RNN as rnn

def networkLoss(v):
  averageLoss = 0
  for i in range(100): # v[0] and v[1] must be whole numbers
    hypotheticalRNN = rnn.RNN(v[0], v[1], v[2])
    hypotheticalRNN.initAdagrad()
    averageLoss += hypotheticalRNN.train(1000)/v[1]
  return averageLoss

def continuousNetworkLoss(v):
  print(v)
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

  def __init__(self, network, tuning_rate = 01e-2, training_steps = 1e3, optimizer=sd.optimize):
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

hyperparameterTuner = HPT(character_model, optimizer = lambda net: sd.optimize
                         (net, 3, maxSteps=20, stepSize=0.05, lowerBound = 1, upperBound = 10, convergence=1e-4, maximize=False))
optimum = hyperparameterTuner.tune()
print("The optimum is %s." %(optimum))
