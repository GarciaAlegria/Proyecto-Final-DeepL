import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision
import os

mixed_precision.set_global_policy('mixed_float16')

devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# Optimize data pipeline
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16  # Reduced from 32
IMG_SIZE = 512   # Reduced from 1024

# Directorio de imágenes
base_dir = 'Images'  # Directorio base donde están los folders de clases

# Crear generadores para los conjuntos de entrenamiento, validación y prueba
train_datagen = ImageDataGenerator(
    rescale=1.0/255, 
    validation_split=0.2, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')  # 20% para validación dentro del entrenamiento
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Generador para el conjunto de entrenamiento
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SIZE, IMG_SIZE),  # Ajusta el tamaño de la imagen aquí
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Para clasificación binaria
    subset='training',
    shuffle=True
)

# Generador para el conjunto de validación
validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# Generador para el conjunto de prueba
test_generator = test_datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Suele ser mejor no barajar para evaluaciones precisas
)


def make_classifier_model():
    model = tf.keras.Sequential()
    
    # Entry block
    model.add(layers.Rescaling(1.0 / 255))
    model.add(layers.Conv2D(128, 3, strides=2, padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    
    # Main blocks
    for size in [256, 512, 728]:
        # First sub-block
        model.add(layers.Activation("relu"))
        model.add(layers.SeparableConv2D(size, 3, padding="same"))
        model.add(layers.BatchNormalization())
        
        # Second sub-block
        model.add(layers.Activation("relu"))
        model.add(layers.SeparableConv2D(size, 3, padding="same"))
        model.add(layers.BatchNormalization())
        
        # Pooling
        model.add(layers.MaxPooling2D(3, strides=2, padding="same"))
    
    # Final conv block
    model.add(layers.SeparableConv2D(1024, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    
    # Classification head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1))  # Binary classification
    
    return model


# Crear el modelo con el número de clases deseado
model = make_classifier_model()

# Compilación del modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',   # para etiquetas enteras (p. ej., 0, 1, 2...)
              metrics=['accuracy'])

# Mostrar un resumen del modelo
model.summary()


# history = model.fit(
#     train_generator,
#     epochs=6,
#     validation_data=validation_generator,
# )

# # Crear el modelo con el número de clases deseado
# model = make_classifier_model()

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Add callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'models/checkpoint.keras',  # Changed from .h5 to .keras
        save_best_only=True
    )
]

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1  # Show progress bar
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# Save the complete model (architecture + weights) in TF format
model.save('saved_models/glasses_detector.keras')

# Optional: Save training history as JSON
import json
if 'history' in locals() or 'history' in globals():
    history_dict = history.history
    with open('saved_models/training_history.json', 'w') as f:
        json.dump(history_dict, f)