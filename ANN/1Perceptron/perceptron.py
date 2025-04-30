import numpy as np

X = np.array([
    [1, -2, 0, 1],   
    [0, 1.5, -0.5, -1],   
    [-1, 1, 0.5, -1]   
])

d = np.array([-1, -1, 1])  


w = np.array([1, -1, 0, 0.5]) 
learning_rate = 0.1


def signum(value):
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0


for i in range(len(X)):
    net = np.dot(w, X[i])  
    print(f"Net: {net}")
    o = signum(net)  
    if o != d[i]:  
        w = w + learning_rate * (d[i] - o) * X[i]
    print(f"Input: {X[i]}, Output: {o}, Updated Weights: {w}")
print("-" * 50)


print("Final Weights:", w)





































# # Function to take input for the array
# def input_array(prompt):
#     print(prompt)
#     rows = int(input("Enter number of rows: "))
#     cols = int(input("Enter number of columns: "))
#     array = []
#     for i in range(rows):
#         row = list(map(float, input(f"Enter row {i+1} values separated by space: ").split()))
#         array.append(row)
#     return np.array(array)

# # Function to take input for the weights
# def input_weights(prompt, size):
#     print(prompt)
#     weights = list(map(float, input(f"Enter {size} weights separated by space: ").split()))
#     return np.array(weights)

# # Taking input from the user
# X = input_array("Enter the input array X:")
# d = np.array(list(map(float, input("Enter the desired output array d separated by space: ").split())))
# w = input_weights("Enter the initial weights w:", X.shape[1])
# learning_rate = float(input("Enter the learning rate: "))