import numpy as np

X = np.array([
    [-1, 1],  # x₁ (Input -1 with bias 1)
    [ 1, 1]   # x₂ (Input 1 with bias 1)
])

Y = np.array([1, -1])  # NOT Gate outputs

W = np.array([0, 0])  # Only two weights (one for input, one for bias)

c = 1

print(f"Initial Weights: {W}")

for i in range(len(X)):
    net_input = np.dot(W, X[i])
    
    y_pred = 0
    if net_input > 0:
        y_pred = 1
    elif net_input < 0:
        y_pred = -1
    else:
        y_pred = 0
    
    W = W + (c * (Y[i] - y_pred) * X[i])

    print(f"After input {i+1}, updated Weights: {W}")

print("\nFinal Weights:", W)