"""
Módulo: calibration_v1.py
Proyecto: Módulo inteligente de calibración ocular basado en EyeGestures
Descripción:
    Este módulo gestiona el proceso de calibración del sistema de seguimiento ocular. 
    Permite establecer puntos de referencia en distintas posiciones de la pantalla 
    (izquierda, derecha, arriba, abajo) y verifica la fijación ocular en cada uno de 
    estos puntos para completar la calibración.

    La calibración se realiza de forma dinámica, adaptando el orden de los puntos 
    según la posición inicial del usuario y agregando pasos adicionales si se detecta 
    un desplazamiento significativo del punto de fijación.

Dependencias:
    - time: para el control de intervalos entre calibraciones.
"""

import time


# ============================================================
# ENUMERACIÓN DE POSICIONES DE CALIBRACIÓN
# ============================================================

class CalibrationPositions:
    """
    Enumeración que define las posibles posiciones de calibración en la pantalla.
    Estas posiciones son utilizadas para guiar al usuario en el proceso de calibración
    de la mirada en diferentes áreas del área de visualización.
    """
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


# ============================================================
# CLASE PRINCIPAL DE CALIBRACIÓN
# ============================================================

class Calibrator:
    """
    Clase encargada de controlar el proceso completo de calibración del sistema ocular.

    Atributos:
        width (int): Ancho total de la pantalla.
        height (int): Alto total de la pantalla.
        start_x (int): Posición inicial del eje X del punto de partida.
        start_y (int): Posición inicial del eje Y del punto de partida.
        calibration_steps (list): Lista que almacena los pasos de calibración pendientes.
        calibration_margin (int): Margen de tolerancia para validar la posición de fijación.
        prev_point (str): Última posición de calibración completada.
        last_calib (float): Marca temporal del último evento de calibración.
    """

    def __init__(self, width, height, start_x, start_y):
        """Inicializa el calibrador con dimensiones de pantalla y posición inicial."""
        self.width = width
        self.height = height
        self.start_x = start_x
        self.start_y = start_y

        self.prev_x = 0
        self.prev_y = 0

        # Margen de tolerancia desde los bordes de pantalla
        self.calibration_margin = 200
        # Lista de pasos de calibración pendientes
        self.calibration_steps = []
        self.__set_order()

        # Indicadores de progreso de calibración
        self.calibrate_left = False
        self.calibrate_right = False
        self.calibrate_top = False
        self.calibrate_bottom = False

        self.calibration = False
        self.drawn = False
        self.prev_point = None
        self.last_calib = time.time()

    # ============================================================
    # MÉTODOS PRIVADOS PARA AGREGAR PASOS
    # ============================================================

    def __add_left(self):
        """Agrega el paso de calibración hacia la izquierda."""
        self.calibration_steps.append(CalibrationPositions.LEFT)
        return self

    def __add_right(self):
        """Agrega el paso de calibración hacia la derecha."""
        self.calibration_steps.append(CalibrationPositions.RIGHT)
        return self

    def __add_top(self):
        """Agrega el paso de calibración hacia arriba."""
        self.calibration_steps.append(CalibrationPositions.TOP)
        return self

    def __add_bottom(self):
        """Agrega el paso de calibración hacia abajo."""
        self.calibration_steps.append(CalibrationPositions.BOTTOM)
        return self

    # ============================================================
    # DEFINICIÓN DEL ORDEN DE CALIBRACIÓN
    # ============================================================

    def __set_order(self):
        """
        Define el orden inicial de los pasos de calibración en función de la posición 
        inicial del usuario en la pantalla (start_x, start_y).

        Si el punto inicial está más cerca de la izquierda, el primer paso será calibrar 
        hacia la izquierda y luego hacia la derecha, y viceversa. Lo mismo ocurre con 
        la parte superior e inferior.
        """
        if self.start_x < self.width / 2 and CalibrationPositions.LEFT not in self.calibration_steps:
            self.__add_left().__add_right()
        elif self.start_x > self.width / 2 and CalibrationPositions.RIGHT not in self.calibration_steps:
            self.__add_right().__add_left()

        if self.start_y < self.height / 2 and CalibrationPositions.TOP not in self.calibration_steps:
            self.__add_top().__add_bottom()
        elif self.start_y > self.height / 2 and CalibrationPositions.BOTTOM not in self.calibration_steps:
            self.__add_bottom().__add_top()

    # ============================================================
    # MANEJO DE RE-CALIBRACIONES
    # ============================================================

    def add_recalibrate(self, recalibrate_step):
        """
        Agrega un paso de recalibración si no está ya en la lista.

        Args:
            recalibrate_step (str): Paso de calibración adicional (LEFT, RIGHT, TOP o BOTTOM).
        """
        if recalibrate_step not in self.calibration_steps:
            self.calibration_steps.append(recalibrate_step)

    # ============================================================
    # PUNTOS DE CALIBRACIÓN ACTUALES
    # ============================================================

    def get_current_point(self):
        """
        Retorna las coordenadas del punto actual de calibración que debe ser mostrado al usuario.

        Returns:
            tuple: Coordenadas (x, y) del punto objetivo actual.
        """
        if len(self.calibration_steps) > 0:
            if CalibrationPositions.LEFT == self.calibration_steps[0]:
                return (self.calibration_margin, int(self.height / 2))
            elif CalibrationPositions.RIGHT == self.calibration_steps[0]:
                return (self.width - self.calibration_margin, int(self.height / 2))
            elif CalibrationPositions.TOP == self.calibration_steps[0]:
                return (int(self.width / 2), self.calibration_margin)
            elif CalibrationPositions.BOTTOM == self.calibration_steps[0]:
                return (int(self.width / 2), self.height - self.calibration_margin)
        else:
            return (0, 0)

    # ============================================================
    # PROCESO PRINCIPAL DE CALIBRACIÓN
    # ============================================================

    def calibrate(self, x, y, fix):
        """
        Controla el proceso de calibración, verificando si el punto actual cumple 
        con los criterios de fijación y posición.

        Args:
            x (float): Coordenada horizontal del punto detectado.
            y (float): Coordenada vertical del punto detectado.
            fix (float): Nivel de fijación o estabilidad (entre 0 y 1).

        Returns:
            bool: True si la calibración del punto fue exitosa, False en caso contrario.
        """
        # Detectar desviaciones grandes y agregar recalibraciones si es necesario
        if abs(x - self.width / 2) > 150 and self.prev_point in [CalibrationPositions.TOP, CalibrationPositions.BOTTOM]:
            if x - self.width / 2 < 0:
                self.add_recalibrate(CalibrationPositions.LEFT)
            else:
                self.add_recalibrate(CalibrationPositions.RIGHT)

        if abs(y - self.height / 2) > 150 and self.prev_point in [CalibrationPositions.LEFT, CalibrationPositions.RIGHT]:
            if y - self.height / 2 < 0:
                self.add_recalibrate(CalibrationPositions.TOP)
            else:
                self.add_recalibrate(CalibrationPositions.BOTTOM)

        # Actualizar posición previa
        self.prev_y = y
        self.prev_x = x

        if len(self.calibration_steps) <= 0:
            self.prev_point = None
            return False

        fixation_thresh = 0.3  # Umbral de fijación mínimo aceptable

        # Verifica si se ha fijado la mirada durante al menos 5 segundos en el punto actual
        if fix > fixation_thresh and (time.time() - self.last_calib) > 5.0:
            # Verifica en qué punto se encuentra el usuario
            if CalibrationPositions.LEFT == self.calibration_steps[0] and x < self.calibration_margin:
                if CalibrationPositions.LEFT in self.calibration_steps:
                    self.calibration_steps.remove(CalibrationPositions.LEFT)
                self.prev_point = CalibrationPositions.LEFT
                self.drawn = False
                self.last_calib = time.time()
                return True

            elif CalibrationPositions.RIGHT == self.calibration_steps[0] and x > self.width - self.calibration_margin:
                if CalibrationPositions.RIGHT in self.calibration_steps:
                    self.calibration_steps.remove(CalibrationPositions.RIGHT)
                self.prev_point = CalibrationPositions.RIGHT
                self.drawn = False
                self.last_calib = time.time()
                return True

            elif CalibrationPositions.TOP == self.calibration_steps[0] and y < self.calibration_margin:
                if CalibrationPositions.TOP in self.calibration_steps:
                    self.calibration_steps.remove(CalibrationPositions.TOP)
                self.prev_point = CalibrationPositions.TOP
                self.drawn = False
                self.last_calib = time.time()
                return True

            elif CalibrationPositions.BOTTOM == self.calibration_steps[0] and y > self.height - self.calibration_margin:
                if CalibrationPositions.BOTTOM in self.calibration_steps:
                    self.calibration_steps.remove(CalibrationPositions.BOTTOM)
                self.prev_point = CalibrationPositions.BOTTOM
                self.drawn = False
                self.last_calib = time.time()
                return True

            # Caso alternativo: corrección del orden de calibración si hay inconsistencias
            else:
                self.last_calib = time.time()
                self.drawn = False
                self.prev_point = None

                if self.calibration_steps[0] in [CalibrationPositions.RIGHT, CalibrationPositions.LEFT]:
                    if x < self.width / 2:
                        if CalibrationPositions.RIGHT in self.calibration_steps:
                            self.calibration_steps.remove(CalibrationPositions.RIGHT)
                        self.calibration_steps.insert(0, CalibrationPositions.RIGHT)
                    else:
                        if CalibrationPositions.LEFT in self.calibration_steps:
                            self.calibration_steps.remove(CalibrationPositions.LEFT)
                        self.calibration_steps.insert(0, CalibrationPositions.LEFT)
                    return True

                if self.calibration_steps[0] is CalibrationPositions.TOP:
                    self.calibration_steps.insert(0, CalibrationPositions.BOTTOM)
                    return True
                else:
                    self.calibration_steps.insert(0, CalibrationPositions.TOP)
                    return True

        # Si no se cumplen condiciones de fijación o tiempo
        self.prev_point = None
        return False

    # ============================================================
    # VERIFICACIÓN DE FINALIZACIÓN
    # ============================================================

    def calibrated(self):
        """
        Indica si el proceso de calibración ha finalizado correctamente.

        Returns:
            bool: True si no quedan pasos pendientes de calibración.
        """
        return len(self.calibration_steps) <= 0
