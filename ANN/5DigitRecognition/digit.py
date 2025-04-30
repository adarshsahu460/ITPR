import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Find one example of each digit (0-9) from test set
digit_examples = {}
for img, label in zip(x_test, y_test):
    if label not in digit_examples:
        digit_examples[label] = img
    if len(digit_examples) == 10:
        break

# Prepare images for prediction
images = np.array([digit_examples[d] for d in range(10)])
predictions = model.predict(images)

# Plot results
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Predicted: {tf.argmax(predictions[i]).numpy()}")
    plt.axis('off')
plt.tight_layout()
plt.show()
