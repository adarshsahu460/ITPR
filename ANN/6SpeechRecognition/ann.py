#  Speech Recognition

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Step 1: Load and Preprocess Data
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Step 2: Load Dataset
data_path = 'recordings/'  # e.g. 0_name_50.wav
labels = []
features = []
file_names = []

for file in os.listdir(data_path):
    if file.endswith('.wav'):
        label = file.split('_')[0]
        labels.append(label)
        feature = extract_features(os.path.join(data_path, file))
        features.append(feature)
        file_names.append(file)  # keep track of filenames

X = np.array(features)
y = np.array(labels)
file_names = np.array(file_names)

# Step 3: Encode Labels
le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    X, y_encoded, file_names, test_size=0.2, random_state=42
)

# Step 5: Build ANN Model
model = Sequential()
model.add(Dense(256, input_shape=(40,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 digits: 0–9

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 6: Train Model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Step 7: Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n🧠 Test Accuracy: {accuracy * 100:.2f}%")

# Step 8: Analyze Predictions
print("\n🔍 Checking individual predictions:")
y_true = le.inverse_transform(np.argmax(y_test, axis=1))
y_pred_probs = model.predict(X_test)
y_pred = le.inverse_transform(np.argmax(y_pred_probs, axis=1))

correct = 0
for i in range(len(y_true)):
    status = "✅" if y_true[i] == y_pred[i] else "❌"
    print(f"{status} File: {filenames_test[i]} | Actual: {y_true[i]} | Predicted: {y_pred[i]}")
    if y_true[i] == y_pred[i]:
        correct += 1

print(f"\n✅ Correct Predictions: {correct}/{len(y_true)} ({(correct/len(y_true))*100:.2f}%)")
