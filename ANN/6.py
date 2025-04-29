import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split

# Function to extract MFCC features from audio files
def extract_mfcc(file_path, max_length=100, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Pad or truncate to fixed length
    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]
    return mfcc

# Load and preprocess Speech Commands dataset
def load_speech_commands(data_dir, max_length=100, n_mfcc=13):
    labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    X, y = [], []
    
    for label_idx, label in enumerate(labels):
        label_dir = os.path.join(data_dir, label)
        if not os.path.exists(label_dir):
            continue
        for file in os.listdir(label_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(label_dir, file)
                mfcc = extract_mfcc(file_path, max_length, n_mfcc)
                X.append(mfcc)
                y.append(label_idx)
    
    X = np.array(X)
    y = np.array(y)
    # Convert labels to one-hot encoding
    y = tf.keras.utils.to_categorical(y, len(labels))
    return X, y, labels

# Placeholder for dataset path (user needs to download Speech Commands dataset)
data_dir = './speech_commands_v0.02'  # Update with actual path

# Load dataset
X, y, class_names = load_speech_commands(data_dir)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape MFCCs for CNN input (add channel dimension)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(13, 100, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Predict on a few test samples
predictions = model.predict(X_test[:5])
predicted_labels = np.argmax(predictions, axis=1)
actual_labels = np.argmax(y_test[:5], axis=1)

# Print predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"Sample {i+1}: Predicted = {class_names[predicted_labels[i]]}, Actual = {class_names[actual_labels[i]]}")