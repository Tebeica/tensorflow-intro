import tensorflow as tf
import numpy as np

print("--Get data--")
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print("--Process data--")
print(len(y_train))
x_train, x_test = x_train / 255.0, x_test / 255.0

print("--Make model--")
# inital model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.1)
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
# model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=2)
model.fit(x_train, y_train, batch_size=128, epochs=30, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuracy: {model_acc * 100:.1f}%")

# model.save("notMNIST.h5")
