import numpy as np

data = open("texts/Shakespeare.txt", "r").read()

chars = list(set(data)) 
data_size, vocab_size = len(data), len(chars)
# print('data has %d chars, %d unique' % (data_size, vocab_size))

char_to_ix = { ch:i for i,ch in enumerate(chars)}
ix_to_char = { i:ch for i, ch in enumerate(chars)}

vector_for_char_a = np.zeros((vocab_size, 1))
vector_for_char_a[char_to_ix['a']] = 1

class RNN:

    def __init__(self, hidden_size = 80, seq_length = 20, learning_rate = 1e-1, init_hidden = True):
      #model parameters
      self.hidden_size = hidden_size
      self.seq_length = seq_length
      self.learning_rate = learning_rate

      if init_hidden:
          self.Wxh = np.random.randn(self.hidden_size, vocab_size) * 0.01 #input to hidden
          self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01 #input to hidden
          self.Why = np.random.randn(vocab_size, self.hidden_size) * 0.01 #input to hidden
          self.bh = np.zeros((self.hidden_size, 1))
          self.by = np.zeros((vocab_size, 1))

    def initAdagrad(self):
      p=0  
      self.inputs = [char_to_ix[ch] for ch in data[p: p + self.seq_length]]
      self.targets = [char_to_ix[ch] for ch in data[p+1: p + self.seq_length + 1]]
      n, p = 0, 0
      self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
      self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad                                                                                                                
      self.smooth_loss = -np.log(1.0/vocab_size)*self.seq_length # loss at iteration 0     

    def lossFun(self, inputs, targets, hprev):
      """                                                                                                                                                                                         
      inputs,targets are both list of integers.                                                                                                                                                   
      hprev is Hx1 array of initial hidden state                                                                                                                                                  
      returns the loss, gradients on model parameters, and last hidden state                                                                                                                      
      """
      #store our inputs, hidden states, outputs, and probability values
      xs, hs, ys, ps, = {}, {}, {}, {} #Empty dicts
      # Each of these are going to be SEQ_LENGTH(Here 25) long dicts i.e. 1 vector per time(seq) step
      # xs will store 1 hot encoded input characters for each of 25 time steps (26, 25 times)
      # hs will store hidden state outputs for 25 time steps (100, 25 times)) plus a -1 indexed initial state
      # to calculate the hidden state at t = 0
      # ys will store targets i.e. expected outputs for 25 times (26, 25 times), unnormalized probabs
      # ps will take the ys and convert them to normalized probab for chars
      # We could have used lists BUT we need an entry with -1 to calc the 0th hidden layer
      # -1 as  a list index would wrap around to the final element
      xs, hs, ys, ps = {}, {}, {}, {}
      #init with previous hidden state
        # Using "=" would create a reference, this creates a whole separate copy
        # We don't want hs[-1] to automatically change if hprev is changed
      hs[-1] = np.copy(hprev)
      #init loss as 0
      loss = 0
      # forward pass                                                                                                                                                                              
      for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation (we place a 0 vector as the t-th input)                                                                                                                     
        xs[t][inputs[t]] = 1 # Inside that t-th input we use the integer in "inputs" list to  set the correct
        hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state                                                                                                            
        ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars                                                                                                           
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
        loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)                                                                                                                       
      # backward pass: compute gradients going backwards    
      #initalize vectors for gradient values for each set of weights 
      dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
      dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
      dhnext = np.zeros_like(hs[0])
      for t in reversed(range(len(inputs))):
        #output probabilities
        dy = np.copy(ps[t])
        #derive our first gradient
        dy[targets[t]] -= 1 # backprop into y  
        #compute output gradient -  output times hidden states transpose
        #When we apply the transpose weight matrix,  
        #we can think intuitively of this as moving the error backward
        #through the network, giving us some sort of measure of the error 
        #at the output of the lth layer. 
        #output gradient
        dWhy += np.dot(dy, hs[t].T)
        #derivative of output bias
        dby += dy
        #backpropagate!
        dh = np.dot(self.Why.T, dy) + dhnext # backprop into h                                                                                                                                         
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity                                                                                                                     
        dbh += dhraw #derivative of hidden bias
        dWxh += np.dot(dhraw, xs[t].T) #derivative of input to hidden layer weight
        dWhh += np.dot(dhraw, hs[t-1].T) #derivative of hidden layer to hidden layer weight
        dhnext = np.dot(self.Whh.T, dhraw) 
      for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients                                                                                                                 
      return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]


    #prediction, one full forward pass
    def sample(self, h, seed_ix, n):
      """                                                                                                                                                                                         
      sample a sequence of integers from the model                                                                                                                                                
      h is memory state, seed_ix is seed letter for first time step   
      n is how many characters to predict
      """
      #create vector
      x = np.zeros((vocab_size, 1))
      #customize it for our seed char
      x[seed_ix] = 1
      #list to store generated chars
      ixes = []
      #for as many characters as we want to generate
      for t in range(n):
        #a hidden state at a given time step is a function 
        #of the input at the same time step modified by a weight matrix 
        #added to the hidden state of the previous time step 
        #multiplied by its own hidden state to hidden state matrix.
        h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
        #compute output (unnormalised)
        y = np.dot(self.Why, h) + self.by
        ## probabilities for next chars
        p = np.exp(y) / np.sum(np.exp(y))
        #pick one with the highest probability 
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        #create a vector
        x = np.zeros((vocab_size, 1))
        #customize it for the predicted char
        x[ix] = 1
        #add it to the list
        ixes.append(ix)

      txt = ''.join(ix_to_char[ix] for ix in ixes)
      open("output.txt", "w").write(txt)
      print('----\n %s \n----' % (txt, ))

    def sampleN(self, n):
      hprev = np.zeros((self.hidden_size,1)) # reset RNN memory  
      #predict the n next characters given 'a'
      self.sample(hprev, char_to_ix['a'], n)

    def train(self, steps):
      ### sanity checks ###
      # Square how mistaken the hyperparameters     #
      # are so second order expansions can operate. #
      if self.learning_rate < 1e-4:
        return -self.learning_rate**2 + 1e6
      if self.learning_rate > 1:
        return self.learning_rate**2 + 1e6
      if self.hidden_size < 20:
          return -self.hidden_size**2 + 1e6
      if self.hidden_size > 800:
          return self.hidden_size**2 + 1e6
      if self.seq_length < 5:
          return -self.seq_length**2 + 1e6
      if self.seq_length > 40:
          return self.seq_length**2 + 1e6
      ### /sanity check ###
      p, n = 0, 0                                                                                                                   
      while n <= steps:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        # check "How to feed the loss function to see how this part works
        if p + self.seq_length+1 >= len(data) or n == 0:
          hprev = np.zeros((self.hidden_size,1)) # reset RNN memory                                                                                                                                      
          p = 0 # go from start of data                                                                                                                                                             
        self.inputs = [char_to_ix[ch] for ch in data[p: p + self.seq_length]]
        self.targets = [char_to_ix[ch] for ch in data[p+1: p + self.seq_length + 1]]
        # forward seq_length characters through the net and fetch gradient                                                                                                                          
        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(self.inputs, self.targets, hprev)
        self.smooth_loss = 0.999 * self.smooth_loss + 0.001 * loss

        # perform parameter update with Adagrad                                                                                                                                                     
        for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
          mem += dparam * dparam
          param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update                                                                                                                   

        p += self.seq_length # move data pointer                                                                                                                                                         
        n += 1 # iteration counter
      return self.smooth_loss
