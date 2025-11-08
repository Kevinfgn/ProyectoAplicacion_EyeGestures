"""
Módulo: heatmap.py
Proyecto: Módulo inteligente de calibración ocular basado en EyeGestures
Descripción:
    Este módulo define la clase `Heatmap`, encargada de representar y procesar 
    el mapa de calor de los puntos rastreados durante la calibración y el seguimiento ocular.

    El propósito del mapa de calor es identificar las zonas donde se concentra la mirada del usuario
    (Región de Interés o ROI), permitiendo ajustar dinámicamente los parámetros del sistema.

Dependencias:
    - numpy
"""

import numpy as np


class Heatmap:
    """
    Clase que representa el mapa de calor generado a partir de puntos rastreados.

    El mapa de calor se utiliza para detectar las regiones con mayor densidad
    de puntos oculares en el proceso de seguimiento y calibración.
    """

    def __init__(self, width, height, buffor):
        """
        Constructor de la clase Heatmap.

        Args:
            width (int): Ancho del área de visualización.
            height (int): Alto del área de visualización.
            buffor (list): Lista de coordenadas (x, y) registradas durante el seguimiento ocular.

        Descripción:
            Divide la pantalla en una cuadrícula de "barras" o secciones
            y acumula la cantidad de puntos que caen dentro de cada celda.
            Luego calcula los límites del área más activa (ROI térmica).
        """
        # Incremento en cada paso del histograma
        self.inc_step = 10
        # Tamaño de los pasos (resolución de la cuadrícula)
        self.step = 10
        bars = self.step

        self.width = width
        self.height = height

        # Cantidad de divisiones en los ejes X e Y
        bars_x = int(width / self.step)
        bars_y = int(height / self.step)

        # Inicializa ejes como vectores de ceros (intensidad por barra)
        self.axis_x = np.zeros((bars_x))
        self.axis_y = np.zeros((bars_y))

        # Recorre los puntos del buffer y acumula sus apariciones por sección
        for point in buffor:
            x = point[0]
            y = point[1]

            self.axis_x[min(abs(int(x / bars)), bars_x - 1)] += self.inc_step
            self.axis_y[min(abs(int(y / bars)), bars_x - 1)] += self.inc_step

        # Determina los límites de la zona de alta densidad de puntos
        self.min_x = self.__getParam((self.axis_x > self.inc_step * 4), last=False)
        self.max_x = self.__getParam((self.axis_x > self.inc_step * 4), last=True)
        self.min_y = self.__getParam((self.axis_y > self.inc_step * 4), last=False)
        self.max_y = self.__getParam((self.axis_y > self.inc_step * 4), last=True)

    def __getParam(self, param, last: bool = False):
        """
        Función privada para obtener el límite (mínimo o máximo) de las zonas activas del heatmap.

        Args:
            param (array): Vector booleano que indica qué secciones superan el umbral de actividad.
            last (bool): Si es True, devuelve el último índice válido; si es False, el primero.

        Returns:
            int: Índice correspondiente al límite detectado, escalado a la resolución del heatmap.
        """
        ret = 0
        retArray = np.where(param)

        if len(retArray[0]) > 0:
            # Selecciona primer o último índice según el parámetro "last"
            ret = retArray[0][-int(last)] * self.step

            # Asegura que el valor no sea NaN
            if not ret == np.nan:
                ret = int(ret)

        return ret

    def getBoundaries(self):
        """
        Retorna los límites del mapa de calor.

        Returns:
            tuple: (x, y, ancho, alto) de la región activa del heatmap.
        """
        x = self.min_x
        y = self.min_y
        w = self.max_x - self.min_x
        h = self.max_y - self.min_y
        return (x, y, w, h)

    def getCenter(self):
        """
        Calcula y retorna el centro de la región activa del mapa de calor.

        Returns:
            tuple: Coordenadas (x, y) del centro del heatmap.
        """
        center_x = int((self.max_x - self.min_x) / 2 + self.min_x)
        center_y = int((self.max_y - self.min_y) / 2 + self.min_y)
        return (center_x, center_y)

    def getPeak(self):
        """
        Retorna el punto de mayor concentración de actividad.

        Returns:
            tuple: Coordenadas (x, y) del pico de densidad de puntos.
        """
        x = int(np.argmax(self.axis_x) * self.inc_step)
        y = int(np.argmax(self.axis_y) * self.inc_step)
        return (x, y)

    def getHist(self):
        """
        Retorna el histograma normalizado de los puntos del heatmap.

        Descripción:
            El heatmap se construye como un histograma 2D,
            por lo que esta función devuelve los valores acumulados
            para los ejes X e Y normalizados por el incremento.

        Returns:
            tuple: (histograma_x, histograma_y)
        """
        return (self.axis_x / self.inc_step, self.axis_y / self.inc_step)
