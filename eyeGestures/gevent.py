"""
Módulo: gevent.py
Descripción general:
    Este módulo define las estructuras de eventos relacionadas con la mirada (gaze events)
    dentro del sistema EyeGestures.  
    Los eventos encapsulan la información procesada por el sistema de seguimiento ocular,
    incluyendo:
      - Coordenadas de la mirada en pantalla.
      - Estado de parpadeo.
      - Nivel de fijación visual.
      - Información de depuración como ROI, bordes y datos de los ojos.

Clases:
    - Gevent: Representa un evento de mirada durante la operación normal del sistema.
    - Cevent: Representa un evento de calibración de mirada.
"""


class Gevent:
    """
    Clase que representa un evento de mirada ("Gaze Event").

    Esta clase almacena los resultados de un cuadro (frame) procesado del sistema
    de seguimiento ocular, incluyendo la posición estimada de la mirada,
    si el usuario ha parpadeado, y el nivel de fijación visual.

    También puede contener datos auxiliares para depuración o análisis, como:
    - Región de interés (ROI).
    - Bordes detectados.
    - Objetos Eye de ambos ojos.
    - Contexto de seguimiento.
    - Información de micro-movimientos o sacadas.

    Atributos:
        point (tuple): Coordenadas (x, y) de la mirada escaladas a la pantalla.
        blink (bool): Indica si se detectó un parpadeo.
        fixation (float): Nivel de fijación (0.0 a 1.0).
        saccades (bool): Indica si se detectó una sacada (movimiento rápido de ojos).

        # Datos auxiliares para depuración
        roi (object): Región de interés (ROI) actual en pantalla.
        edges (object): Información sobre los bordes detectados.
        l_eye (Eye): Objeto representando el ojo izquierdo.
        r_eye (Eye): Objeto representando el ojo derecho.
        cluster (object): Grupo de puntos relacionados con la mirada actual.
        context (object): Contexto activo del seguimiento ocular.
        screen_man (object): Administrador de pantalla que gestiona el mapeo.
        sub_frame (ndarray | None): Subimagen opcional del cuadro procesado.
    """

    def __init__(self,
                 point,
                 blink,
                 fixation,
                 l_eye=None,
                 r_eye=None,
                 screen_man=None,
                 roi=None,
                 edges=None,
                 cluster=None,
                 context=None,
                 saccades=False,
                 sub_frame=None):

        # Datos principales del evento
        self.point = point
        self.blink = blink
        self.fixation = fixation
        self.saccades = saccades

        # Datos adicionales para análisis o depuración
        self.roi = roi
        self.edges = edges
        self.l_eye = l_eye
        self.r_eye = r_eye
        self.cluster = cluster
        self.context = context
        self.screen_man = screen_man
        self.sub_frame = sub_frame


class Cevent:
    """
    Clase que representa un evento de calibración ("Calibration Event").

    Esta clase se utiliza durante el proceso de calibración del sistema de
    seguimiento ocular. Contiene la posición del punto de mirada detectado
    y los radios de aceptación y calibración, que definen las tolerancias
    del sistema al validar la precisión del seguimiento.

    Atributos:
        point (tuple): Coordenadas (x, y) estimadas de la mirada.
        acceptance_radius (float): Radio de tolerancia para aceptar una muestra.
        calibration_radius (float): Radio que define el área de calibración.
        calibration (bool): Indica si el evento ocurre durante una sesión de calibración.
    """

    def __init__(self,
                 point,
                 acceptance_radius,
                 calibration_radius,
                 calibration=False):

        self.point = point
        self.acceptance_radius = acceptance_radius
        self.calibration_radius = calibration_radius
        self.calibration = calibration
