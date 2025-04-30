import numpy as np


X = np.array([
    [1, -2, 1.5, 0], 
    [1, -0.5, -2, -1.5], 
    [0, 1, -1, 1.5]
])

w = np.array([1, -1, 0, 0.5]) 

c = 1

def signum(value):
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0

# Hebbian Learning
for i in range(len(X)):
    net = np.dot(w, X[i])
    print(f"Net: {net}")
    
    o = signum(net)  
    w = w + (c * o * X[i])  
    print(f"Input: {X[i]}, Output: {o}, Updated Weights: {w}")

print("-" * 50)
print("Final Weights:", w)












