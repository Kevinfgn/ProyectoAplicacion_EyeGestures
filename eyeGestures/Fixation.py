"""
Módulo: fixation.py
Descripción general:
    Este módulo proporciona una clase para la detección de fijaciones oculares
    a partir de puntos estimados de mirada (x, y).
    Una fijación ocurre cuando la posición del ojo se mantiene estable dentro
    de un radio predefinido durante un periodo de tiempo.

Clases:
    - Fixation: Encapsula la lógica de detección y persistencia de una fijación.

Dependencias:
    No requiere librerías externas adicionales.
"""


class Fixation:
    """
    Clase encargada de detectar y calcular el nivel de fijación ocular en base
    a las coordenadas de seguimiento (tracking) del ojo.

    La detección se basa en comparar la distancia entre el punto actual y el punto
    previo de fijación. Si el ojo se mantiene dentro de un radio determinado, se 
    considera que hay fijación y esta aumenta progresivamente.

    Atributos:
        radius (float): Radio de tolerancia para considerar una fijación estable.
        fixation (float): Nivel de fijación actual (entre 0.0 y 1.0).
        x (float): Coordenada X del último punto fijado.
        y (float): Coordenada Y del último punto fijado.
    """

    def __init__(self, x, y, radius=100):
        """
        Inicializa un objeto de tipo Fixation con un punto inicial y un radio.

        Args:
            x (float): Coordenada X inicial del punto de fijación.
            y (float): Coordenada Y inicial del punto de fijación.
            radius (float, opcional): Radio de tolerancia para la detección de fijación. 
                                      Por defecto es 100 píxeles.
        """
        self.radius = radius       # Radio máximo permitido para considerar fijación
        self.fixation = 0.0        # Nivel inicial de fijación (sin fijación)
        self.x = x                 # Posición X actual
        self.y = y                 # Posición Y actual
        pass

    def process(self, x, y):
        """
        Procesa un nuevo punto (x, y) y actualiza el estado de fijación.
        Si el punto se encuentra dentro del radio definido, se incrementa
        el valor de fijación; si no, se reinicia.

        Args:
            x (float): Coordenada X del nuevo punto detectado.
            y (float): Coordenada Y del nuevo punto detectado.

        Returns:
            float: Nivel de fijación actual, donde 1.0 representa fijación completa
                   y 0 indica pérdida de fijación.
        """
        # Calcula si el punto actual se encuentra dentro del área de fijación
        if (x - self.x)**2 + (y - self.y)**2 < self.radius**2:
            # Incrementa la fijación gradualmente hasta un máximo de 1.0
            self.fixation = min(self.fixation + 0.02, 1.0)
        else:
            # Si se sale del radio, se reinicia el punto de referencia
            self.x = x
            self.y = y
            self.fixation = 0

        return self.fixation
