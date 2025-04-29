import numpy as np

# Hebbian Network class
class HebbianNetwork:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights (including bias) with zeros
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
    
    def activation(self, x):
        # Bipolar activation function (-1 or 1)
        return 1 if x >= 0 else -1
    
    def predict(self, inputs):
        # Add bias input (1)
        inputs_with_bias = np.append(inputs, 1)
        # Calculate weighted sum
        weighted_sum = np.dot(inputs_with_bias, self.weights)
        return self.activation(weighted_sum)
    
    def train(self, training_inputs, epochs):
        for epoch in range(epochs):
            for inputs in training_inputs:
                # Convert inputs to bipolar (-1, 1)
                bipolar_inputs = np.where(inputs > 0, 1, -1)
                # Add bias input (1)
                inputs_with_bias = np.append(bipolar_inputs, 1)
                # Get output
                output = self.predict(bipolar_inputs)
                # Update weights using Hebbian rule: w_new = w_old + lr * x * y
                self.weights += self.learning_rate * inputs_with_bias * output
            # Print epoch results
            print(f"Epoch {epoch + 1}, Weights: {self.weights}")

# Training data: 3 inputs (using bipolar values for Hebbian learning)
training_inputs = np.array([
    [1, 1, 1],   # Positive pattern
    [-1, -1, -1], # Negative pattern
    [1, -1, 1],  # Mixed pattern
    [-1, 1, -1]  # Mixed pattern
])

# Create and train Hebbian network
network = HebbianNetwork(input_size=3)
print("Initial Weights:", network.weights)
network.train(training_inputs, epochs=10)

# Test the trained network
print("\nTesting trained network:")
for inputs in training_inputs:
    prediction = network.predict(inputs)
    print(f"Input: {inputs}, Predicted Output: {prediction}")