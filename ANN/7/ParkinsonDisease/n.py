import pandas as pd #load the data set
import numpy as np # perform the operation 
from sklearn.model_selection import train_test_split  # split the data set
from sklearn.preprocessing import StandardScaler # to scal the data set
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import tensorflow as tf

# Set the seed for Python's random module
random.seed(42) 

# Set the seed for NumPy
np.random.seed(42)

# Set the seed for TensorFlow
tf.random.set_seed(42)

# Read the dataset
df = pd.read_csv("./Parkinsson disease.csv")  # Replace with correct path

# Drop non-numeric columns (e.g., 'name')
df = df.drop(['name'], axis=1)

# Features and target
X = df.drop(['status'], axis=1)  # Features predictors
y = df['status']                 # Target (0: healthy, 1: Parkinson's) 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print(X_test_scaled)
# Build ANN model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, verbose=0)
loss, accuracy1 = model.evaluate(X_train_scaled, y_train,verbose=0)
print(f"Train Accuracy: {accuracy1:.2f}")


# Evaluate
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")



# Prediction for new data (example)

def predict_parkinsons(new_input):
    """
    Predict Parkinson's based on new voice features input
    :param new_input: list of features (in same order as dataset)
    """
    new_input_scaled = scaler.transform([new_input])
    prediction = model.predict(new_input_scaled)[0][0]
    if prediction > 0.5:
        return "Likely Parkinson's"
    else:
        return "Likely Healthy"

# Example usage:
# new_data_sample = list(X.iloc[0])  # Just for testing


# new_data_sample = [
#     119.992, 157.302, 74.997, 0.00784, 0.00007, 0.00370, 0.00554, 0.01109,
#     0.04374, 0.426, 0.02182, 0.03130, 0.02971, 0.06545,
#     0.02211, 21.033, 0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654
# ]

new_data_sample = [
    237.226, 247.326, 225.227, 0.00298, 0.00001, 0.00169, 0.00182, 0.00507,
    0.01752, 0.164, 0.01035, 0.01024, 0.01133, 0.03104,
    0.0074, 22.736, 0.305062, 0.654172, -7.31055, 0.098648, 2.416838, 0.095032
]

# new_data_sample = [
#     116.014, 141.781, 110.655, 0.01284, 0.00011, 0.00655, 0.00908, 0.01966,
#     0.06425, 0.584, 0.0349, 0.04825, 0.04465, 0.1047,
#     0.01767, 19.649, 1, 0.417356, 0.823484, -3.747787, 0.234513, 2.33218, 0.410335
# ]

# new_data_sample = [
#     245.51, 262.09, 231.848, 0.00235, 0.00001, 0.00127, 0.00148, 0.0038,
#     0.01608, 0.141, 0.00906, 0.00977, 0.01149, 0.02719,
#     0.00476, 24.602, 0, 0.467489, 0.631653, -7.156076, 0.127642, 2.392122, 0.097336
# ]



print(predict_parkinsons(new_data_sample))
