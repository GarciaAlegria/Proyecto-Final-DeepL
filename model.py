import os # Importamos la librería os para poder crear directorios
import json # Importamos la librería json para poder guardar el historial de entrenamiento
import tensorflow as tf # Importamos la librería tensorflow
from tensorflow.keras import layers # Importamos la clase layers de la librería tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Importamos la clase ImageDataGenerator de la librería tensorflow.keras
from tensorflow.keras import mixed_precision # Importamos la clase mixed_precision de la librería tensorflow.keras

mixed_precision.set_global_policy('mixed_float16') # Establecemos la política de precisión mixta en mixed_float16

devices = tf.config.list_physical_devices() # Obtenemos los dispositivos físicos disponibles
print("\nDevices: ", devices) # Imprimimos los dispositivos físicos disponibles

gpus = tf.config.list_physical_devices('GPU') # Obtenemos las GPUs disponibles en caso de que existan
if gpus: # Si existen GPUs
    for gpu in gpus: # Iteramos sobre las GPUs
        tf.config.experimental.set_memory_growth(gpu, True) # Establecemos el crecimiento de la memoria en True

AUTOTUNE = tf.data.AUTOTUNE # Establecemos la variable AUTOTUNE con el valor de tf.data.AUTOTUNE
BATCH_SIZE = 16 # Establecemos el tamaño del lote en 16
IMG_SIZE = 512 # Establecemos el tamaño de la imagen en 512

base_dir = 'Images' # Establecemos el directorio base en 'Images'

train_datagen = ImageDataGenerator( # Creamos un objeto de la clase ImageDataGenerator para el conjunto de entrenamiento
    rescale=1.0/255,  # Normalizamos los valores de los píxeles
    validation_split=0.2,  # Establecemos el porcentaje de validación en 20%
    rotation_range=20, # Establecemos el rango de rotación en 20
    width_shift_range=0.2, # Establecemos el rango de desplazamiento horizontal en 0.2
    height_shift_range=0.2, # Establecemos el rango de desplazamiento vertical en 0.2
    shear_range=0.2, # Establecemos el rango de cizallamiento en 0.2
    zoom_range=0.2, # Establecemos el rango de zoom en 0.2
    horizontal_flip=True, # Establecemos el volteo horizontal en True
    fill_mode='nearest') # Establecemos el modo de relleno en 'nearest'
test_datagen = ImageDataGenerator(rescale=1.0/255) # Creamos un objeto de la clase ImageDataGenerator para el conjunto de prueba

train_generator = train_datagen.flow_from_directory( # Creamos un generador de flujo de datos para el conjunto de entrenamiento
    base_dir, # Establecemos el directorio base
    target_size=(IMG_SIZE, IMG_SIZE), # Establecemos el tamaño de la imagen
    batch_size=BATCH_SIZE, # Establecemos el tamaño del lote en 16
    class_mode='binary', # Establecemos el modo de clasificación en 'binary'
    subset='training', # Establecemos el subconjunto en 'training'
    shuffle=True # Establecemos el barajado en True
)

validation_generator = train_datagen.flow_from_directory( # Creamos un generador de flujo de datos para el conjunto de validación
    base_dir, # Establecemos el directorio base
    target_size=(IMG_SIZE, IMG_SIZE), # Establecemos el tamaño de la imagen
    batch_size=BATCH_SIZE, # Establecemos el tamaño del lote en 16
    class_mode='binary', # Establecemos el modo de clasificación en 'binary'
    subset='validation', # Establecemos el subconjunto en 'validation'
    shuffle=True # Establecemos el barajado en True
)

test_generator = test_datagen.flow_from_directory( # Creamos un generador de flujo de datos para el conjunto de prueba
    base_dir, # Establecemos el directorio base
    target_size=(IMG_SIZE, IMG_SIZE), # Establecemos el tamaño de la imagen
    batch_size=BATCH_SIZE, # Establecemos el tamaño del lote en 16
    class_mode='binary', # Establecemos el modo de clasificación en 'binary'
    shuffle=False # Establecemos el barajado en False
)


def make_classifier_model(): # Definimos la función make_classifier_model para crear el modelo clasificador
    model = tf.keras.Sequential() # Creamos un modelo secuencial
    
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(512, 512, 3))) # Añadimos una capa convolucional con 64 filtros, un tamaño de kernel de 3x3, un relleno de 'same' y una forma de entrada de 512x512x3
    model.add(layers.BatchNormalization()) # Añadimos una capa de normalización por lotes
    model.add(layers.LeakyReLU(alpha=0.1)) # Añadimos una capa de activación LeakyReLU con un alpha de 0.1
    model.add(layers.MaxPooling2D((2, 2))) # Añadimos una capa de agrupación máxima con un tamaño de 2x2
    model.add(layers.Dropout(0.3)) # Añadimos una capa de abandono con una tasa de abandono de 0.3
    
    model.add(layers.Conv2D(128, (3, 3), padding='same')) # Añadimos una capa convolucional con 128 filtros y un tamaño de kernel de 3x3
    model.add(layers.BatchNormalization()) # Añadimos una capa de normalización por lotes
    model.add(layers.LeakyReLU(alpha=0.1)) # Añadimos una capa de activación LeakyReLU con un alpha de 0.1
    model.add(layers.MaxPooling2D((2, 2))) # Añadimos una capa de agrupación máxima con un tamaño de 2x2
    model.add(layers.Dropout(0.3)) # Añadimos una capa de abandono con una tasa de abandono de 0.3
    
    model.add(layers.Conv2D(256, (3, 3), padding='same')) # Añadimos una capa convolucional con 256 filtros y un tamaño de kernel de 3x3
    model.add(layers.BatchNormalization()) # Añadimos una capa de normalización por lotes
    model.add(layers.LeakyReLU(alpha=0.1)) # Añadimos una capa de activación LeakyReLU con un alpha de 0.1
    model.add(layers.MaxPooling2D((2, 2))) # Añadimos una capa de agrupación máxima con un tamaño de 2x2
    model.add(layers.Dropout(0.3)) # Añadimos una capa de abandono con una tasa de abandono de 0.3
    
    model.add(layers.Conv2D(512, (3, 3), padding='same')) # Añadimos una capa convolucional con 512 filtros y un tamaño de kernel de 3x3
    model.add(layers.BatchNormalization()) # Añadimos una capa de normalización por lotes
    model.add(layers.LeakyReLU(alpha=0.1)) # Añadimos una capa de activación LeakyReLU con un alpha de 0.1 
    model.add(layers.MaxPooling2D((2, 2))) # Añadimos una capa de agrupación máxima con un tamaño de 2x2
    model.add(layers.Dropout(0.3)) # Añadimos una capa de abandono con una tasa de abandono de 0.3
    
    model.add(layers.Flatten()) # Añadimos una capa de aplanamiento
    model.add(layers.Dense(256)) # Añadimos una capa densa con 256 unidades
    model.add(layers.BatchNormalization()) # Añadimos una capa de normalización por lotes 
    model.add(layers.LeakyReLU(alpha=0.1)) # Añadimos una capa de activación LeakyReLU con un alpha de 0.1
    model.add(layers.Dropout(0.5)) # Añadimos una capa de abandono con una tasa de abandono de 0.5
    model.add(layers.Dense(1, activation='sigmoid', dtype='float32')) # Añadimos una capa densa con 1 unidad y una función de activación sigmoide
    
    return model # Retornamos el modelo


model = make_classifier_model() # Creamos el modelo clasificador

model.compile(optimizer='adam', # Compilamos el modelo con el optimizador Adam
              loss='binary_crossentropy', # Establecemos la función de pérdida en binary_crossentropy
              metrics=['accuracy']) # Establecemos la métrica en precisión 

model.summary() # Imprimimos un resumen del modelo para visualizar su arquitectura
 
if not os.path.exists('models'): # Si no existe el directorio 'models' 
    os.makedirs('models') # Creamos el directorio 'models'

callbacks = [ # Definimos una lista de callbacks
    tf.keras.callbacks.ReduceLROnPlateau( # Añadimos un callback para reducir la tasa de aprendizaje en caso de que la pérdida de validación no mejore
        monitor='val_loss', # Establecemos la métrica a monitorear en 'val_loss'
        factor=0.2, # Establecemos el factor de reducción en 0.2
        patience=2 # Establecemos la paciencia en 2
    ),
    tf.keras.callbacks.ModelCheckpoint( # Añadimos un callback para guardar el modelo en caso de que la precisión de validación mejore
        'models/checkpoint.keras', # Establecemos el nombre del archivo de control en 'models/checkpoint.keras'
        save_best_only=True # Guardamos solo el mejor modelo
    )
]

history = model.fit( # Entrenamos el modelo con el generador de flujo de datos de entrenamiento y validación
    train_generator, # Establecemos el generador de flujo de datos de entrenamiento
    epochs=25, # Establecemos el número de épocas en 25
    validation_data=validation_generator, # Establecemos el generador de flujo de datos de validación
    callbacks=callbacks, # Establecemos los callbacks
    verbose=1 # Establecemos el nivel de verbosidad en 1
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size) # Evaluamos el modelo con el generador de flujo de datos de prueba

print(f"Test Accuracy: {test_accuracy * 100:.2f}%") # Imprimimos la precisión de prueba en porcentaje
print(f"Test Loss: {test_loss:.4f}") # Imprimimos la pérdida de prueba con 4 decimales


if not os.path.exists('saved_models'): # Si no existe el directorio 'saved_models'
    os.makedirs('saved_models') # Creamos el directorio 'saved_models'

model.save('saved_models/glasses_detector.keras') # Guardamos el modelo en el archivo 'saved_models/glasses_detector.keras'

if 'history' in locals() or 'history' in globals(): # Si la variable 'history' existe en el ámbito local o global
    history_dict = history.history # Obtenemos el historial de entrenamiento
    with open('saved_models/training_history.json', 'w') as f: # Abrimos el archivo 'saved_models/training_history.json' en modo de escritura
        json.dump(history_dict, f) # Guardamos el historial de entrenamiento en el archivo 'saved_models/training_history.json'