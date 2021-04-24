import numpy as np

# 1. Coding a Neuron
inputs = [1, 2, 3, 2.5]  # outputs from previous 3 neurons
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2  # bias is a constant which helps the model in a way that it can fit best for the given data

output1 = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + bias

print(output1)

# 2.Modelling 3 Neurons with 4 inputs --> a layer

inputs = [1, 2, 3, 2.5]  # outputs from previous 3 neurons
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2  # bias is a constant which helps the model in a way that it can fit best for the given data
bias2 = 3
bias3 = 0.5

output = [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
          inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,
          inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3]
print(output)

# 3. Code above, but generalised
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

layer_outputs = []  # output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)

# 4.Dot Product with numpy
inputs4 = [1, 2, 3, 2.5]
weights4 = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
biases4 = [2, 3, 0.5]
output4 = np.dot(weights4, inputs4) + biases4
print(output4)

# 5. Dot product with transpose; Batches, Layers, Objects
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
output = np.dot(inputs, np.array(weights).T) + biases4
print(output)
# add a new layer to the nn
biases2 = [-1, 2, -0.5]
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, 0.33],
            [-0.44, 0.73, -0.13]]
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases4
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)

# 6.Softmax activation function
print('Softmax activation function')
layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)

# 7.Calculating loss with categorical cross-entropy
import math

print("Loss")
softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])
print(loss)
loss = -math.log(softmax_output[0])
print(loss)
