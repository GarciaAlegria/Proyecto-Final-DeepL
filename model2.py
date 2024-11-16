import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  details = tf.config.experimental.get_device_details(gpus[0])
  print("GPU details: ", details)


# Directorio de imágenes
base_dir = 'Images'  # Directorio base donde están los folders de clases

# Crear generadores para los conjuntos de entrenamiento, validación y prueba
train_datagen = ImageDataGenerator(
    rescale=1.0/255, )  # 20% para validación dentro del entrenamiento
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Generador para el conjunto de entrenamiento
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(512, 512),  # Ajusta el tamaño de la imagen aquí
    batch_size=32,
    class_mode='binary',  # Para clasificación binaria
    subset='training',
    shuffle=True
)

# Generador para el conjunto de validación
validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(512, 512),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# Generador para el conjunto de prueba
test_generator = test_datagen.flow_from_directory(
    base_dir,
    target_size=(512, 512),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Suele ser mejor no barajar para evaluaciones precisas
)


def make_classifier_model():
    model = tf.keras.Sequential()
    
    # First Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(512, 512, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Second Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Third Convolutional Block
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Fourth Convolutional Block
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid', dtype='float32'))  # Sigmoid activation for binary classification
    
    return model


# Crear el modelo con el número de clases deseado
model = make_classifier_model()

# Compilación del modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',   # para etiquetas enteras (p. ej., 0, 1, 2...)
              metrics=['accuracy'])

# Mostrar un resumen del modelo
model.summary()


history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator,
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Create directory if it doesn't exist
import os
if not os.path.exists('models'):
    os.makedirs('models')

# Save the complete model (architecture + weights) in TF format
model.save('models/glasses_detector')

# Optional: Save training history as JSON
import json
if 'history' in locals() or 'history' in globals():
    history_dict = history.history
    with open('models/training_history.json', 'w') as f:
        json.dump(history_dict, f)