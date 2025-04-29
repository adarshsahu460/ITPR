import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size + 1)
        self.learning_rate = learning_rate
    
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        inputs_with_bias = np.append(inputs, 1)
        weighted_sum = np.dot(inputs_with_bias, self.weights)
        return self.activation(weighted_sum)
    
    def train(self, training_inputs, labels, epochs):
        for epoch in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                inputs_with_bias = np.append(inputs, 1)
                self.weights += self.learning_rate * error * inputs_with_bias
            print(f"Epoch {epoch + 1}, Weights: {self.weights}")

training_inputs = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

labels = np.array([0, 0, 0, 0, 0, 0, 0, 1])

perceptron = Perceptron(input_size=3)
print("Initial Weights:", perceptron.weights)
perceptron.train(training_inputs, labels, epochs=10)

print("\nTesting trained perceptron:")
for inputs in training_inputs:
    prediction = perceptron.predict(inputs)
    print(f"Input: {inputs}, Predicted Output: {prediction}")