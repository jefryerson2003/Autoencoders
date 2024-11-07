import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Carga de la base de datos Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 2. Preprocesamiento de los datos
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((x_train.shape[0], 784))
x_test = x_test.reshape((x_test.shape[0], 784))

# 3. Construcción del autoencoder para visualizar en 2D
input_dim = 784
encoding_dim = 2  # Dimensión para visualización en 2D

# Definición del modelo de autoencoder
input_img = layers.Input(shape=(input_dim,))
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(encoding_dim, activation='linear')(encoded)  # Capa de codificación en 2D
decoded = layers.Dense(128, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = models.Model(input_img, decoded)
encoder = models.Model(input_img, encoded)

# 4. Compilación y entrenamiento del autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# 5. Visualización en 2D de los datos de prueba usando el autoencoder
encoded_imgs = encoder.predict(x_test)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, cmap='tab10')
plt.colorbar(scatter)
plt.title('Visualización 2D de Fashion MNIST usando Autoencoder')
plt.show()

# 6. Visualización en 2D de los datos de prueba usando PCA
pca = PCA(n_components=2)
x_test_pca = pca.fit_transform(x_test)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=y_test, cmap='tab10')
plt.colorbar(scatter)
plt.title('Visualización 2D de Fashion MNIST usando PCA')
plt.show()

# 7. Verificación de la capacidad de generación de nuevas imágenes
decoder_input = layers.Input(shape=(encoding_dim,))
decoder_layer1 = autoencoder.layers[-2](decoder_input)
decoder_layer2 = autoencoder.layers[-1](decoder_layer1)
decoder = models.Model(decoder_input, decoder_layer2)

# Generar puntos aleatorios en el espacio latente
random_points = np.random.normal(size=(10, encoding_dim))

generated_images = decoder.predict(random_points)
generated_images = generated_images.reshape((10, 28, 28))

# Mostrar las imágenes generadas
plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(generated_images[i], cmap='gray')
    plt.axis('off')
plt.suptitle('Imágenes generadas por el Autoencoder')
plt.show()
