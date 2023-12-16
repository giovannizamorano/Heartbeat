#carpetas en las que se subiran las imagenes de prueba
# N-Normal
# S-Latido supraventricular prematuro
# V-Contraccion Ventricular prematura
# F-Fusion de latido ventricular y normal
# Q-Latidos no clasificados
# M-Infarto agudo de miocardio
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2

#plt.figure(figsize=(15,15))
#carpeta = 'C:/Users/Giovanni/OneDrive/Escritorio/Heartbeat/dataset/Contraccion_ventricular_prematura'
#imagenes = os.listdir(carpeta)

#for i, nombreimg in enumerate(imagenes[:25]):

#  plt.subplot(5,5,i+1)
#   imagen = mpimg.imread(carpeta + '/' + nombreimg)
#plt.imshow(imagen)

#normalizacion de las imagenes
datagen_entrenamiento = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range = 30,
    width_shift_range = 0.25,
    height_shift_range = 0.25,
    shear_range = 15,
    zoom_range = [0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
)

datagen_pruebas = ImageDataGenerator(rescale=1. / 255) 

data_gen_entrenamiento = datagen_entrenamiento.flow_from_directory('C:/Users/Giovanni/OneDrive/Escritorio/Heartbeat/dataset', target_size=(224,224),
                                                     batch_size=32, shuffle=True)

data_gen_pruebas = datagen_pruebas.flow_from_directory('C:/Users/Giovanni/OneDrive/Escritorio/Heartbeat/datatest', target_size=(224,224),
                                                     batch_size=32, shuffle=True)


modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), input_shape=(224,224,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), #tamaño de la matriz

    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), #tamaño de la matriz

    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), #tamaño de la matriz

    tf.keras.layers.Conv2D(256,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), #tamaño de la matriz

    tf.keras.layers.Conv2D(512,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), #tamaño de la matriz

    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1024, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
    ])
#compilador del modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

TAMANO_LOTE = 300

print("Entrenando modelo...")
epocas=2500
historial = modelo.fit(
    data_gen_entrenamiento,
    epochs=epocas,
    batch_size=TAMANO_LOTE,
    validation_data=data_gen_pruebas,
    steps_per_epoch=int(np.ceil(len(data_gen_entrenamiento) / float(TAMANO_LOTE))),
    validation_steps=int(np.ceil(len(data_gen_pruebas) / float(TAMANO_LOTE)))
)

print("modelo entrenado")



#Graficas de precision
acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']
loss=historial.history['loss']
val_loss=historial.history['val_loss']
rango_epocas = range(epocas)


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(rango_epocas, acc, label='Precision Entrenamiento')
plt.plot(rango_epocas, val_acc, label='Precision Pruebas')
plt.legend(loc='lower right')
plt.title('precision de entrenamiento y pruebas')

plt.subplot(1,2,2)
plt.plot(rango_epocas, loss, label='Perdida Entrenamiento')
plt.plot(rango_epocas, val_loss, label='Perdida de Pruebas')
plt.legend(loc='upper right')
plt.title('perdida de entrenamiento y pruebas')
plt.show()


def categorizar(url):
  respuesta = requests.get(url)
  img = Image.open(BytesIO(respuesta.content))
  img = np.array(img).astype(float)/255
  if len(img.shape) == 2:
    if img.dtype == np.float64:
      img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

  img = cv2.resize(img,(224,224))

  if img.shape[-1] == 4:
    img = img[:, :, :3]

  img = np.expand_dims(img, axis=0)
  prediccion = modelo.predict(img)
  return np.argmax(prediccion[0], axis=-1)

# 0 = Contraccion_ventricular_prematura, 1 = Fusion_latido_ventricular_y_normal, 2= Infarto_agudo_al_miocardio, 3= Latidos_sin_clasificar,4= Normal, 5= Supraventricular_prematuro


modelo.save('heartbeat.h5')
'''
urls = ['https://i.postimg.cc/JhPQJsDM/N54.png', 'https://i.postimg.cc/mkHMtsk4/M0.png', 'https://i.postimg.cc/rwC11MJF/F15.png', 'https://i.postimg.cc/Y9VYr9p8/F30.png','https://i.postimg.cc/8c6WW7kw/N25.png','https://i.postimg.cc/jd6nb0DW/Q33.png','https://i.postimg.cc/13Jg8Gcs/S22.png','https://i.postimg.cc/brxrpkwX/S38.png','https://i.postimg.cc/sf7vYd9n/S55.png','https://i.postimg.cc/t4CJG10f/V11.png','https://i.postimg.cc/rFTzgnfq/V6.png']

for url in urls:
  print(url)
  prediccion = categorizar (url)
  if prediccion == 0:
    print("Contraccion_ventricular_prematura")
    print(prediccion)

  elif prediccion == 1:
    print("Fusion_latido_ventricular_y_normal")
    print(prediccion)

  elif prediccion == 2:
    print("Infarto_agudo_al_miocardio")
    print(prediccion)

  elif prediccion == 4:
    print("Normal")
    print(prediccion)

  elif prediccion == 5:
    print("Supraventricular_prematuro")
    print(prediccion)

  else:
    print("Latidos_sin_clasificar")
    print(prediccion)
'''