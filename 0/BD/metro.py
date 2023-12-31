# %%
!unzip icons.zip

# %%
import cv2
import numpy as np
import os

carpeta_imagenes = 'icons'
datos_imagenes = []

for nombre_archivo in os.listdir(carpeta_imagenes):
    ruta_imagen = os.path.join(carpeta_imagenes, nombre_archivo)
    imagen = cv2.imread(ruta_imagen)
    arreglo_numeros = np.array(imagen)
    datos_imagenes.append({'nombre': nombre_archivo, 'arreglo_numeros': arreglo_numeros})

# %%
datos_imagenes

# %%
ruta_nueva_imagen = 'IMG_1173.png'
nueva_imagen = cv2.imread(ruta_nueva_imagen)

nueva_imagen_redimensionada = cv2.resize(nueva_imagen, (133, 133))

arreglo_nueva_imagen = np.array(nueva_imagen_redimensionada)

resultados_correlacion = {}

for datos_imagen in datos_imagenes:
    nombre_imagen = datos_imagen['nombre']
    arreglo_imagen_guardada = datos_imagen['arreglo_numeros']

    resultado_correlacion = cv2.matchTemplate(arreglo_nueva_imagen, arreglo_imagen_guardada, cv2.TM_CCOEFF_NORMED)

    resultados_correlacion[nombre_imagen] = resultado_correlacion

valor_max_correlacion = max(resultados_correlacion.values())

imagenes_con_max_correlacion = [nombre_imagen for nombre_imagen, correlacion in resultados_correlacion.items() if correlacion == valor_max_correlacion]

print("El valor máximo de correlación es:", valor_max_correlacion)
print("Las imágenes con este valor son:")
for nombre_imagen in imagenes_con_max_correlacion:
    print(nombre_imagen)


