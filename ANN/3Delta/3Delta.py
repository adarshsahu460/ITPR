import numpy as np


X = np.array([
    [1, -2, 0, -1], 
    [0, 1.5, -0.5, -1], 
    [-1, 1, 0.5, -1]
])

d = np.array([-1, -1, 1]) 


w = np.array([1, -1, 0, 0.5])  


c = 0.1  

l = 1

def sigmoid(x):
    return 2 / (1 + np.exp(-x)) - 1  


def sigmoid_derivative(y):
    return 0.5 * (1 - y**2) 


for i in range(len(X)):
    net = np.dot(w, X[i]) 
    o = sigmoid(l*net) 
    
    f_prime = sigmoid_derivative(o)  

    w = w + c * (d[i] - o) * f_prime * X[i]

    print(f"Iteration {i+1}:")
    print(f"Net: {net:.2f}, Output: {o:.2f}")
    print(f"Updated Weights: [{', '.join(f'{weight:.2f}' for weight in w)}]")
    print("-" * 50)

print("Final Weights:", [f"{weight:.2f}" for weight in w])





