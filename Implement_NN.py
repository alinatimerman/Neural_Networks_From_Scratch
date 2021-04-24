import numpy as np
import nnfs
from nnfs import spiral_data_generator
np.random.seed(0)

class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights=0.10*np.random.rand(n_inputs, n_neurons) #randn gaussian distribution bounded to zero
        self.biases=np.zeros((1, n_neurons))                    #diff shape for weights --> no transpose
    def forward(self, inputs):
        self.output=np.dot(inputs, self.weights)+self.biases

class Activation_ReLU:   #step function==> transforms any negative value into a zero
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probabilities=exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output=probabilities

X,y=spiral_data(samples=100, classes=3)
dense1=Layer_Dense(2,3)
activation1=Activation_ReLU()

dense2=Layer_Dense(3,3)
activation2=Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])


layer1= Layer_Dense(4,5)
activation1=Activation_ReLU()
layer1.forward(X)
print('layer before activation:')
print(layer1.output)
activation1.forward(layer1.output)
print('after activation')
print(activation1.output)

#layer2= Layer_Dense(5,2)
#layer2.forward(layer1.output)
#print(layer2.output)





