import numpy as np

X = np.array([
    [-1, -1, 1],  # x₁
    [-1,  1, 1],  # x₂
    [ 1, -1, 1],  # x₃
    [ 1,  1, 1]   # x₄
])

Y = np.array([-1, -1, -1, 1])  

W = np.array([0, 0, 0])

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