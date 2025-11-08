"""
Módulo: processing.py
Descripción general:
    Este módulo implementa la clase `EyeProcessor`, encargada de transformar
    la posición de la pupila en coordenadas de dirección de mirada (gaze direction).

    La clase procesa los datos de las características del ojo (landmarks) y la posición
    de la pupila detectada para escalar, normalizar y promediar las coordenadas
    dentro de una región de interés. Este procesamiento es esencial para
    obtener una estimación precisa de la dirección de la mirada.

Clases:
    - EyeProcessor: Gestiona el procesamiento de las coordenadas de la pupila
      y su conversión a un sistema de referencia escalado.
"""

import numpy as np


class EyeProcessor:
    """
    Clase que procesa la posición de la pupila y la traduce a dirección de mirada.

    Esta clase normaliza las coordenadas de la pupila dentro del área del ojo,
    aplica escalado para adaptarlas a la resolución de la pantalla de seguimiento,
    y mantiene un historial (buffer) de posiciones para calcular un punto promedio estable.

    Atributos:
        scale_w (int): Ancho de escala del área procesada.
        scale_h (int): Alto de escala del área procesada.
        min_x (float): Límite mínimo del ojo en el eje X.
        max_x (float): Límite máximo del ojo en el eje X.
        min_y (float): Límite mínimo del ojo en el eje Y.
        max_y (float): Límite máximo del ojo en el eje Y.
        pupil (tuple): Última posición detectada de la pupila.
        landmarks (np.ndarray): Coordenadas de los puntos característicos del ojo.
    """

    def __init__(self, scale_w=250, scale_h=250):
        self.scale_w = scale_w
        self.scale_h = scale_h
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0
        self.pupil = None
        self.landmarks = None

    def append(self, pupil: (int, int), landmarks: np.ndarray, pupilBuffor):
        """
        Agrega un nuevo punto de pupila y lo transforma al sistema escalado.

        Este método actualiza los límites del ojo según los puntos faciales detectados
        y añade al buffer una versión escalada y normalizada de la posición de la pupila.

        Args:
            pupil (tuple[int, int]): Coordenadas (x, y) de la pupila detectada.
            landmarks (np.ndarray): Conjunto de puntos del ojo detectados por MediaPipe.
            pupilBuffor (Buffor): Objeto que almacena un historial de posiciones de pupila.
        """

        self.pupil = pupil
        self.landmarks = landmarks

        # Determinar el área del ojo con un margen de seguridad
        margin = 5
        self.min_x = np.min(self.landmarks[:, 0]) - margin
        self.max_x = np.max(self.landmarks[:, 0]) + margin
        self.min_y = np.min(self.landmarks[:, 1]) - margin
        self.max_y = np.max(self.landmarks[:, 1]) + margin

        # Validar que el punto de pupila esté dentro de la región del ojo
        assert self.pupil[0] > self.min_x
        assert self.pupil[1] > self.min_y

        width = self.max_x - self.min_x
        height = (self.max_y - self.min_y) / 2  # proporción para mejor estabilidad

        # Agregar punto convertido al buffer
        pupilBuffor.add(
            self.__convertPoint(
                self.pupil,
                width=self.scale_w,
                height=self.scale_h,
                scale_w=width,
                scale_h=height,
                offset=(self.min_x, self.min_y)
            )
        )

    def __convertPoint(self,
                       point,
                       width=1.0,
                       height=1.0,
                       scale_w=1.0,
                       scale_h=1.0,
                       offset=(0.0, 0.0)):
        """
        Convierte un punto del espacio original a un sistema de coordenadas escalado.

        Args:
            point (tuple): Coordenadas originales (x, y) de la pupila.
            width (float): Ancho del área escalada.
            height (float): Alto del área escalada.
            scale_w (float): Escala horizontal aplicada.
            scale_h (float): Escala vertical aplicada.
            offset (tuple): Desplazamiento (x, y) del área del ojo.

        Returns:
            tuple[int, int]: Coordenadas (x, y) escaladas del punto.
        """
        (min_x, min_y) = offset
        x = int(((point[0] - min_x) / scale_w) * width)
        y = int(((point[1] - min_y) / scale_h) * height)
        return (x, y)

    def getAvgPupil(self, width, height, pupilBuffor):
        """
        Obtiene el punto promedio de la pupila en el sistema escalado.

        Usa los valores almacenados en el buffer para suavizar variaciones
        y devolver una posición promedio representativa del punto de mirada.

        Args:
            width (float): Ancho de la pantalla de referencia.
            height (float): Alto de la pantalla de referencia.
            pupilBuffor (Buffor): Objeto que almacena los puntos recientes de la pupila.

        Returns:
            tuple[int, int]: Coordenadas promedio escaladas de la pupila.
        """

        if width is not None and height is not None:
            _retPupil = self.__convertPoint(
                pupilBuffor.getAvg(),
                width=width,
                height=height,
                scale_w=self.scale_w,
                scale_h=self.scale_h
            )
        else:
            _retPupil = pupilBuffor.getAvg()

        return _retPupil
