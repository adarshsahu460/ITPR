import numpy as np

# Delta Network class
class DeltaNetwork:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights (including bias) with random values
        self.weights = np.random.rand(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
    
    def activation(self, x):
        # Linear activation function (λ = 1 as per requirement)
        return x
    
    def predict(self, inputs):
        # Add bias input (1)
        inputs_with_bias = np.append(inputs, 1)
        # Calculate weighted sum
        weighted_sum = np.dot(inputs_with_bias, self.weights)
        return self.activation(weighted_sum)
    
    def train(self, training_inputs, labels, epochs):
        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(training_inputs, labels):
                # Get prediction
                prediction = self.predict(inputs)
                # Calculate error
                error = label - prediction
                total_error += error ** 2
                # Update weights using Delta rule: w_new = w_old + c * error * x
                inputs_with_bias = np.append(inputs, 1)
                self.weights += self.learning_rate * error * inputs_with_bias
            # Print epoch results
            print(f"Epoch {epoch + 1}, Weights: {self.weights}, Mean Squared Error: {total_error / len(training_inputs)}")

# Training data: 3 inputs and corresponding continuous outputs
# Example: Learning a linear function (e.g., sum of inputs)
training_inputs = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [0.2, 0.3, 0.4]
])

# Target outputs (e.g., sum of inputs as a simple target function)
labels = np.array([
    0.6,  # 0.1 + 0.2 + 0.3
    1.5,  # 0.4 + 0.5 + 0.6
    2.4,  # 0.7 + 0.8 + 0.9
    0.9   # 0.2 + 0.3 + 0.4
])

# Create and train Delta network
network = DeltaNetwork(input_size=3, learning_rate=0.1)
print("Initial Weights:", network.weights)
network.train(training_inputs, labels, epochs=10)

# Test the trained network
print("\nTesting trained network:")
for inputs, label in zip(training_inputs, labels):
    prediction = network.predict(inputs)
    print(f"Input: {inputs}, Predicted Output: {prediction:.4f}, Target: {label}")