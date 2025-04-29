import numpy as np

# Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights (including bias) with random values
        self.weights = np.random.rand(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
    
    def activation(self, x):
        # Step function for binary output
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        # Add bias input (1)
        inputs_with_bias = np.append(inputs, 1)
        # Calculate weighted sum
        weighted_sum = np.dot(inputs_with_bias, self.weights)
        return self.activation(weighted_sum)
    
    def train(self, training_inputs, labels, epochs):
        for epoch in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                # Get prediction
                prediction = self.predict(inputs)
                # Calculate error
                error = label - prediction
                # Update weights
                inputs_with_bias = np.append(inputs, 1)
                self.weights += self.learning_rate * error * inputs_with_bias
            # Print epoch results
            print(f"Epoch {epoch + 1}, Weights: {self.weights}")

# Training data: 2 inputs for logical AND and OR
training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target outputs for AND
and_labels = np.array([0, 0, 0, 1])

# Target outputs for OR
or_labels = np.array([0, 1, 1, 1])

# Train perceptron for AND function
print("Training Perceptron for AND function")
and_perceptron = Perceptron(input_size=2)
print("Initial Weights:", and_perceptron.weights)
and_perceptron.train(training_inputs, and_labels, epochs=10)

# Test AND perceptron
print("\nTesting AND Perceptron:")
for inputs in training_inputs:
    prediction = and_perceptron.predict(inputs)
    print(f"Input: {inputs}, Predicted Output: {prediction}")

# Train perceptron for OR function
print("\nTraining Perceptron for OR function")
or_perceptron = Perceptron(input_size=2)
print("Initial Weights:", or_perceptron.weights)
or_perceptron.train(training_inputs, or_labels, epochs=10)

# Test OR perceptron
print("\nTesting OR Perceptron:")
for inputs in training_inputs:
    prediction = or_perceptron.predict(inputs)
    print(f"Input: {inputs}, Predicted Output: {prediction}")