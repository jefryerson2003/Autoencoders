import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.utils import to_categorical

# Cargar y preprocesar los datos
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape((x_train.shape[0], 784))
x_test = x_test.reshape((x_test.shape[0], 784))
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Definir el autoencoder
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

# Compilar y entrenar el autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Usar el encoder para extracción de características
x_train_encoded = encoder.predict(x_train)
x_test_encoded = encoder.predict(x_test)

# Definir y entrenar el clasificador
classifier = Sequential([
    Dense(64, activation='relu', input_shape=(64,)),
    Dense(10, activation='softmax')
])

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(x_train_encoded, y_train, epochs=50, batch_size=256, validation_data=(x_test_encoded, y_test))

# Evaluación del modelo
test_loss, test_acc = classifier.evaluate(x_test_encoded, y_test)
print("Accuracy en test:", test_acc)
