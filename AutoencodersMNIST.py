import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. Carga de la base de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocesamiento de los datos
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((x_train.shape[0], 784))
x_test = x_test.reshape((x_test.shape[0], 784))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 3. Construcción del autoencoder apilado
input_dim = 784  # Tamaño de entrada de la imagen (28x28)
encoding_dim = 64  # Dimensión de la capa latente

# Definición del encoder
input_img = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)

# Definición del decoder
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

# Modelo completo del autoencoder
autoencoder = models.Model(input_img, decoded)
encoder = models.Model(input_img, encoded)  # Modelo solo del encoder

# 4. Compilación del autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# 5. Entrenamiento del autoencoder
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# 6. Construcción del clasificador
encoded_input = layers.Input(shape=(encoding_dim,))
classifier_output = layers.Dense(10, activation='softmax')(encoded_input)
classifier = models.Model(encoded_input, classifier_output)

# Congelar las capas del encoder para usarlo sin ajuste
for layer in encoder.layers:
    layer.trainable = False

# 7. Compilación del clasificador
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 8. Generar las características latentes
encoded_train = encoder.predict(x_train)
encoded_test = encoder.predict(x_test)

# 9. Entrenamiento del clasificador
classifier.fit(encoded_train, y_train, epochs=50, batch_size=256, validation_split=0.2)

# 10. Evaluación del clasificador
loss, accuracy = classifier.evaluate(encoded_test, y_test)
print(f"Precisión del clasificador con autoencoder: {accuracy:.2%}")
