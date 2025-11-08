"""
Módulo: csv_reader.py
Descripción general:
    Este módulo permite leer y deserializar datos de seguimiento ocular (gaze data)
    almacenados en archivos CSV. Los datos incluyen información sobre el punto de mirada,
    eventos de parpadeo, fijación y las posiciones detectadas de los ojos (landmarks y pupilas).

    El proceso de lectura convierte cadenas serializadas (almacenadas en el CSV)
    en estructuras de datos reales utilizando `pickle`. Este tipo de procesamiento es
    común en sistemas de seguimiento ocular (eye-tracking) para análisis posterior o
    reconstrucción de sesiones de calibración.

Dependencias:
    - pickle: para deserializar los objetos almacenados como texto.
    - csv: para la lectura secuencial del archivo CSV.
    - ast: para evaluar expresiones en formato de cadena (si se usara en versiones previas).

Uso típico:
    Se llama a la función `read_gaze_data_from_csv(filename)` proporcionando el nombre
    del archivo CSV que contiene los datos del rastreador ocular.  
    La función devuelve una lista de diccionarios con las variables deserializadas.

Ejemplo de uso:
    >>> filename = 'collection_91675a1d.csv'
    >>> gaze_data = read_gaze_data_from_csv(filename)
    >>> for data in gaze_data:
    >>>     print(data)
"""

import pickle
import csv
import ast


def read_gaze_data_from_csv(filename):
    """
    Lee datos de eventos de mirada (gaze events) desde un archivo CSV.

    Args:
        filename (str): Nombre del archivo CSV a leer.

    Returns:
        list: Lista de diccionarios, donde cada diccionario contiene información
              relacionada con un evento de mirada, incluyendo coordenadas, fijaciones
              y datos de landmarks oculares.

    Estructura esperada del CSV:
        Cada fila (excepto el encabezado) debe contener:
            [0] point_x            → Coordenada X del punto de mirada
            [1] point_y            → Coordenada Y del punto de mirada
            [2] blink              → Indicador de parpadeo
            [3] fixation           → Nivel de fijación (valor flotante)
            [4] screen_x           → Posición X en pantalla
            [5] screen_y           → Posición Y en pantalla
            [6] l_eye_landmarks    → Datos serializados de landmarks del ojo izquierdo
            [7] r_eye_landmarks    → Datos serializados de landmarks del ojo derecho
            [8] l_eye_pupil        → Datos serializados de la pupila izquierda
            [9] r_eye_pupil        → Datos serializados de la pupila derecha
            [10] screen_width      → Ancho de la pantalla
            [11] screen_height     → Altura de la pantalla
            [12] rois              → Regiones de interés serializadas (ROI)

    Detalles de implementación:
        - Utiliza `pickle.loads()` para reconstruir los objetos guardados como texto.
        - Emplea la codificación `latin-1` y `unicode_escape` para manejar los bytes serializados.
        - Imprime temporalmente las filas leídas (para depuración).
    """

    gaze_data = []

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for n, row in enumerate(reader):
            # Se omite la primera fila (encabezado)
            if n > 0:
                print(n, row[6], type(row[6]), row[6][2])
                print(bytearray(f'{row[6][2:-3]}', 'latin-1'))

                # Deserialización de campos serializados con pickle
                row_data = {
                    "point_x": row[0],
                    "point_y": row[1],
                    "blink": row[2],
                    "fixation": row[3],
                    "screen_x": row[4],
                    "screen_y": row[5],
                    "l_eye_landmarks": pickle.loads(
                        bytearray(f'{row[6][2:-1]}', 'latin-1')
                        .decode('unicode_escape')
                        .encode('latin1')
                    ),
                    "r_eye_landmarks": pickle.loads(
                        bytearray(f'{row[7][2:-1]}', 'latin-1')
                        .decode('unicode_escape')
                        .encode('latin1')
                    ),
                    "l_eye_pupil": pickle.loads(
                        bytearray(f'{row[8][2:-1]}', 'latin-1')
                        .decode('unicode_escape')
                        .encode('latin1')
                    ),
                    "r_eye_pupil": pickle.loads(
                        bytearray(f'{row[9][2:-1]}', 'latin-1')
                        .decode('unicode_escape')
                        .encode('latin1')
                    ),
                    "screen_width": row[10],
                    "screen_height": row[11],
                    "rois": row[12],
                }

                # Se agrega el evento procesado a la lista general
                gaze_data.append(row_data)

    return gaze_data


# Ejemplo de uso práctico
if __name__ == "__main__":
    filename = 'collection_91675a1d.csv'
    gaze_data = read_gaze_data_from_csv(filename)
    for data in gaze_data:
        print(data)
