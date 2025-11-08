"""
Módulo: dataPoints.py
Proyecto: Módulo inteligente de calibración ocular basado en EyeGestures
Descripción:
    Este módulo define estructuras auxiliares (helpers) para representar los 
    distintos elementos involucrados en el seguimiento ocular y la calibración 
    de pantalla, como el centro de un objeto, el área visible de la pantalla (ROI), 
    el display y la pantalla completa.

    Su propósito es modelar los componentes espaciales (coordenadas, tamaños, 
    posiciones) que se utilizan para calcular la posición del ojo o del punto de 
    mirada dentro del sistema EyeGestures.

Dependencias:
    - numpy
    - scipy.signal
"""

import math
import numpy as np
from scipy import signal


class Center:
    """
    Clase auxiliar que representa el centro de un objeto rectangular.

    Se utiliza para obtener las coordenadas del punto central de un objeto
    definido por sus coordenadas y dimensiones.
    """

    def __init__(self, x, y, width, height):
        """
        Constructor de la clase Center.

        Args:
            x (float): Coordenada X del objeto.
            y (float): Coordenada Y del objeto.
            width (float): Ancho del objeto.
            height (float): Alto del objeto.
        """
        # Calcula el punto central del rectángulo
        self.x = (x + width) / 2
        self.y = (y + height) / 2


class Screen:
    """
    Clase que representa la pantalla principal (Screen).

    Define un rectángulo base de dimensiones fijas (ancho, alto) que 
    sirve como referencia para el sistema de seguimiento ocular.
    """

    def __init__(self, width, height):
        """
        Constructor de la clase Screen.

        Args:
            width (float): Ancho total de la pantalla.
            height (float): Alto total de la pantalla.
        """
        self.x = 0
        self.y = 0
        self.width = width
        self.height = height

    def getCenter(self):
        """
        Retorna el centro de la pantalla.

        Returns:
            Center: Objeto Center con las coordenadas del centro de la pantalla.
        """
        return Center(self.x, self.y, self.width, self.height)


class ScreenROI:
    """
    Clase que representa una Región de Interés (ROI) dentro de la pantalla.

    La ROI se utiliza para delimitar el área donde se analiza la mirada o 
    se realiza la calibración. Permite modificar su posición y tamaño.
    """

    def __init__(self, x, y, width, height):
        """
        Constructor de la clase ScreenROI.

        Args:
            x (float): Coordenada X inicial.
            y (float): Coordenada Y inicial.
            width (float): Ancho del área de interés.
            height (float): Alto del área de interés.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def setCenter(self, x, y):
        """
        Permite mover la ROI cambiando su centro.

        Args:
            x (float): Nueva coordenada X del centro.
            y (float): Nueva coordenada Y del centro.
        """
        self.x = x - self.width / 2
        self.y = y - self.height / 2

    def getCenter(self):
        """
        Retorna el centro actual de la ROI.

        Returns:
            Center: Objeto con las coordenadas del centro.
        """
        return Center(self.x, self.y, self.width, self.height)

    def getBoundaries(self):
        """
        Retorna los límites (frontera rectangular) de la ROI.

        Returns:
            tuple: (x, y, ancho, alto)
        """
        return (self.x, self.y, self.width, self.height)


class Display:
    """
    Clase que representa el Display físico donde se proyectan las coordenadas del seguimiento ocular.

    Este objeto considera las dimensiones físicas del display y un desplazamiento
    (offset) que permite ubicarlo correctamente respecto al sistema de referencia.
    """

    def __init__(self, width, height, offset_x, offset_y):
        """
        Constructor de la clase Display.

        Args:
            width (float): Ancho físico del display.
            height (float): Alto físico del display.
            offset_x (float): Desplazamiento en eje X respecto al origen.
            offset_y (float): Desplazamiento en eje Y respecto al origen.
        """
        self.width = width
        self.height = height
        self.offset_x = offset_x
        self.offset_y = offset_y

        # Se podría utilizar un buffer para almacenar muestras, 
        # pero en esta versión está comentado.
        # self.buffor = Buffor(20)

    def getCenter(self):
        """
        Retorna el centro del display.

        Returns:
            Center: Objeto Center con las coordenadas del centro del display.
        """
        return self.__center
