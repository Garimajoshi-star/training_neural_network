# Import libraries
import numpy as np

# Normalising input data
xAll = np.array(([2, 5, 4], [1, 6, 7], [3,4,6], [2,6,5],[3,5, 7]), dtype=int) 
y = np.array(([75], [82], [96], [81]), dtype=int) 
xAll = xAll/np.amax(xAll, axis=0) 
y = y/100 

X = np.split(xAll, [4])[0
xPredicted = np.split(xAll, [4])[1] 

y = np.array(([75], [82], [96],[81]), dtype=int)
y = y/100 

# Define Neural Network class
class Neural_Network(object):
  def __init__(self):
    self.inputSize = 3
    self.outputSize = 1
    self.hiddenSize = 3

# Generate random weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 
  def forward(self, X):
    self.z = np.dot(X, self.W1) 
    self.z2 = self.sigmoid(self.z) 
    self.z3 = np.dot(self.z2, self.W2) 
    o = self.sigmoid(self.z3) 
    return o

# Define sigmoid function

def sigmoid(self, s):
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    return s * (1 - s)
# def backward(self, X, y, o):
    
    self.o_error = y - o 
    self.o_delta = self.o_error*self.sigmoidPrime(o) 

    self.z2_error = self.o_delta.dot(self.W2.T) 
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) 

    self.W1 += X.T.dot(self.z2_delta) 
    self.W2 += self.z2.T.dot(self.o_delta) 

# Define training data 
  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

def predict(self):
    print ("Predicted data based on trained weights: ")
    print ("Input (scaled):"+str(xPredicted))
    print ("Output:" + str(self.forward(xPredicted)))

NN = Neural_Network()

# Predict output value
for i in range(3): 
  print ("# " + str(i) + "\n")
  print ("Predicted Output:" + str(NN.forward(X)))
  print ("Loss:" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  print ("\n")
  NN.train(X, y)

NN.saveWeights()
NN.predict()

