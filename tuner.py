import numpy as np
import optimization.multivariable_calculus as mvc
import optimization.steepest_descent as sd
import optimization.newton as newton
import optimization.quasi_newton as qn
import RNN as rnn


def networkLoss(v):
  hypotheticalRNN = rnn.RNN(60, 20, v[0])
  return hypotheticalRNN.train(10)

class HPT: # HyperparameterTuner

  def __init__(self, network, tuning_rate = 0.01, training_steps = 1000, optimizer=sd.optimize):
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
    # apply Gradient Descent on randomly initialized network hyperparameters. (Tune a hyperparameter tuner??)
    # note: RNN.train returns the loss of the network following the training.
    # This loss, however, increases as the network improves. Therefore, add the gradient to maximize.
    return self.optimizer(networkLoss, 1, maxSteps=40, lowerBound = 0, stepSize=0.01)

character_model = rnn.RNN()
# character_model.train(1000)
# character_model.sampleN(200)

hyperparameterTuner = HPT(character_model, optimizer=sd.optimize)
optimum = hyperparameterTuner.tune()
