"""
Módulo: heatmap_maker.py
Descripción general:
    Este módulo permite generar un mapa de calor (heatmap) basado en datos de seguimiento ocular
    (gaze tracking) obtenidos desde un archivo CSV. El mapa de calor representa las zonas de la pantalla
    donde el usuario ha fijado la mirada con mayor frecuencia o duración.

    Opcionalmente, el mapa de calor puede superponerse a una imagen de fondo (background)
    correspondiente a la escena observada por el usuario.

Dependencias:
    - matplotlib.pyplot: para graficar los mapas de calor e imágenes.
    - numpy: para manipular los datos numéricos y generar los histogramas 2D.
    - argparse: para manejar parámetros desde la línea de comandos.
    - csv, os, pickle, re, ast: para manejo de archivos, deserialización y procesamiento de datos.

Basado en:
    Procesos de visualización de datos de eye-tracking aplicados a la interfaz visual de
    sistemas de interacción humano-computadora (HCI) o análisis de atención visual.

Normas aplicables:
    - IEEE Std 1012-1998: Estándar de verificación y validación de software.
    - IEEE Std 829-1998: Estándar para documentación de pruebas.
    - ASTM E1340-96: Guía estándar para prototipado rápido de sistemas computarizados.

Uso típico:
    Desde la línea de comandos:
        $ python heatmap_maker.py data/collection_xxxx/data.csv --background --window 100 --step 50

    O directamente en un script Python:
        >>> from heatmap_maker import main
        >>> main("data/collection_xxxx/data.csv", background_file=True)
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import csv
import ast
import re
import os


def read_gaze_data_from_csv(filename):
    """
    Lee datos de eventos de mirada (gaze events) desde un archivo CSV.

    Args:
        filename (str): Nombre del archivo CSV que contiene los datos de seguimiento ocular.

    Returns:
        list: Lista de diccionarios, donde cada diccionario contiene:
            - timestamp: Marca temporal del evento.
            - point_x, point_y: Coordenadas crudas del punto de mirada.
            - blink: Indicador de parpadeo.
            - fixation: Nivel de fijación (valor flotante).
            - screen_x, screen_y: Coordenadas escaladas en la pantalla.
            - screen_width, screen_height: Dimensiones de la pantalla.
            - rois: Regiones de interés (ROI) registradas.

    Detalles:
        El archivo CSV debe tener encabezado y las filas siguientes con los datos en el orden descrito.
        No se deserializan estructuras complejas, ya que solo se necesitan los valores numéricos
        para generar el mapa de calor.
    """

    gaze_data = []

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for n, row in enumerate(reader):
            if n > 0:  # Se omite el encabezado
                row_data = {
                    "timestamp": row[0],
                    "point_x": row[1],
                    "point_y": row[2],
                    "blink": row[3],
                    "fixation": row[4],
                    "screen_x": row[5],
                    "screen_y": row[6],
                    "screen_width": row[7],
                    "screen_height": row[8],
                    "rois": row[12]
                }
                gaze_data.append(row_data)

    return gaze_data


def draw_heatmap(data_points, x_dim, y_dim, picture_path=None):
    """
    Dibuja un mapa de calor 2D basado en las coordenadas de los puntos de mirada.

    Args:
        data_points (np.ndarray): Arreglo Nx2 con las coordenadas (x, y) de los puntos.
        x_dim (float): Ancho de la pantalla o área visualizada.
        y_dim (float): Altura de la pantalla o área visualizada.
        picture_path (str, opcional): Ruta a la imagen de fondo sobre la que se superpone el heatmap.

    Detalles:
        - Se utiliza un histograma 2D con 50 bins para generar la densidad de puntos.
        - Si se pasa una imagen, se aplica una transparencia (alpha) para visualizarla junto al heatmap.
    """

    x_range = [0, x_dim]
    y_range = [0, y_dim]

    # Generar el histograma bidimensional de densidad de puntos
    heatmap, xedges, yedges = np.histogram2d(
        data_points[:, 0],
        data_points[:, 1],
        bins=50,
        range=[x_range, y_range]
    )

    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    alpha = 1.0

    plt.clf()

    # Si se proporciona una imagen de fondo, cargarla
    if picture_path:
        map_image = plt.imread(picture_path)
        map_image = np.flipud(map_image)
        alpha = 0.5
        plt.imshow(map_image, extent=extent, aspect='auto')

    # Graficar el heatmap
    plt.imshow(heatmap.T, extent=extent, origin='lower', alpha=alpha, cmap='hot')
    plt.gca().invert_yaxis()
    plt.show()


def main(gaze_data_file, background_file=False, step=None, window_size=None, start=None, stop=None):
    """
    Función principal para generar el mapa de calor completo o por ventanas de tiempo.

    Args:
        gaze_data_file (str): Ruta al archivo CSV con los datos de mirada.
        background_file (bool): Si True, intenta superponer las imágenes de fondo del experimento.
        step (int): Paso entre ventanas de tiempo (en número de muestras).
        window_size (int): Tamaño de la ventana de análisis (en número de muestras).
        start (float): Marca temporal de inicio (opcional).
        stop (float): Marca temporal de finalización (opcional).

    Comportamiento:
        - Si no se especifica `window_size`, se genera un mapa de calor total.
        - Si se especifica `window_size` y `step`, se generan heatmaps por intervalos.
        - Si `background_file` es True, se superpone la imagen correspondiente al timestamp.
    """

    recordings_path = f"{os.path.dirname(gaze_data_file)}/recordings"

    gaze_data = read_gaze_data_from_csv(gaze_data_file)
    data_points = []
    timestamps = []

    screen_w = float(gaze_data[0]["screen_width"])
    screen_h = float(gaze_data[0]["screen_height"])

    for data in gaze_data:
        timestamps.append(float(data["timestamp"]))
        data_points.append([float(data["screen_x"]), float(data["screen_y"])])

    data_points = np.array(data_points)

    # Recortar por tiempo de inicio
    if start:
        index = timestamps.index(start)
        timestamps = timestamps[index:]
        data_points = data_points[index:, :]

    # Recortar por tiempo de finalización
    if stop:
        index = timestamps.index(stop)
        timestamps = timestamps[:index]
        data_points = data_points[:index, :]

    # Sin ventana de tiempo: generar un solo heatmap
    if window_size is None or step is None:
        picture_path = None
        if background_file:
            picture_path = f"{recordings_path}/{timestamps[0]}.png"
        draw_heatmap(data_points, screen_w, screen_h, picture_path)
    else:
        # Generar heatmaps por ventanas deslizantes
        for i in range(0, len(timestamps), step):
            picture_path = None
            if background_file:
                print(f"{background_file}")
                picture_path = f"{recordings_path}/{timestamps[i]}.png"
            data_points_range = data_points[i:i + window_size]
            draw_heatmap(data_points_range, screen_w, screen_h, picture_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Genera un mapa de calor basado en datos de seguimiento ocular sobre una imagen.'
    )
    parser.add_argument('path_to_gaze_data', type=str, help='Ruta al archivo CSV con datos de mirada.')
    parser.add_argument('--window', type=int, help='Tamaño de la ventana en muestras.')
    parser.add_argument('--step', type=int, help='Paso entre ventanas en muestras.')
    parser.add_argument('--background', action='store_true', help='Superponer imagen de fondo.')
    parser.add_argument('--start', type=float, help='Tiempo de inicio.')
    parser.add_argument('--stop', type=float, help='Tiempo de finalización.')

    args = parser.parse_args()

    main(
        args.path_to_gaze_data,
        background_file=args.background,
        step=args.step,
        window_size=args.window,
        start=args.start,
        stop=args.stop
    )
