"""
Módulo principal del paquete eyeGestures
========================================

Este módulo inicializa e integra las distintas versiones del sistema de seguimiento ocular
(Eye Gestures v1, v2 y v3), que permiten detectar y procesar el movimiento de los ojos,
realizar calibraciones dinámicas, y estimar la dirección y fijación de la mirada.

Incluye:
    - Detección facial y ocular mediante `FaceFinder` y `Face`.
    - Estimación de la mirada con `GazeTracker`.
    - Procesamiento de eventos de mirada (`Gevent`, `Cevent`).
    - Mecanismos de calibración (v1 y v2).
    - Gestión de contexto, buffers y filtrado de señal.
    - Soporte para calibración automática y seguimiento en tiempo real.

Basado en técnicas de:
    - Procesamiento de imágenes en OpenCV.
    - Filtrado de señales (Fourier).
    - Seguimiento de landmarks faciales y predicción de posición ocular.

Autoría y uso:
    - Proyecto de Ingeniería en Computadores.
    - Compatible con Python 3.x y OpenCV ≥ 4.0.

Normas aplicadas:
    - IEEE Std 1012: Verificación y validación de software.
    - ISO/IEC 25010: Calidad del producto software.
"""

from eyeGestures.face import FaceFinder, Face
from eyeGestures.Fixation import Fixation
from eyeGestures.gazeEstimator import GazeTracker
import eyeGestures.screenTracker.dataPoints as dp
from eyeGestures.calibration_v1 import Calibrator as Calibrator_v1
from eyeGestures.calibration_v2 import Calibrator as Calibrator_v2
from eyeGestures.gevent import Gevent, Cevent
from eyeGestures.utils import timeit, Buffor, low_pass_filter_fourier, recoverable
import numpy as np
import pickle
import time
import cv2

VERSION = "3.0.0"


# ==============================================================
# EyeGestures v3 — Versión avanzada de seguimiento ocular
# ==============================================================

class EyeGestures_v3:
    """
    Clase principal de la versión 3 del sistema EyeGestures.

    Gestiona:
        - Procesamiento de imagen facial y ocular.
        - Detección de fijación, parpadeos y sacádicos.
        - Calibración adaptativa con puntos de referencia.
        - Seguimiento continuo con reducción de ruido mediante Fourier.

    Args:
        calibration_radius (int): Radio de calibración inicial en píxeles.

    Atributos principales:
        clb (dict): Calibradores activos por contexto.
        finder (FaceFinder): Detector facial.
        face (Face): Objeto que contiene landmarks faciales.
        fixationTracker (dict): Detectores de fijación por contexto.
        velocity_max/min (dict): Registro de velocidades para detectar sacádicos.
    """

    def __init__(self, calibration_radius=1000):
        self.calibration_radius = calibration_radius
        self.clb = dict()
        self.cap = None

        # Estados de calibración y promedios
        self.calibration = dict()
        self.average_points = dict()
        self.iterator = dict()
        self.filled_points = dict()
        self.enable_CN = False
        self.calibrate_gestures = False

        # Componentes faciales
        self.finder = FaceFinder()
        self.face = Face()

        # Estados dinámicos de contexto
        self.prev_timestamp = dict()
        self.prev_point = dict()
        self.fix = dict()
        self.velocity_max = dict()
        self.velocity_min = dict()
        self.fixationTracker = dict()
        self.key_points_buffer = dict()

        # Posición base de la cabeza
        self.starting_head_position = np.zeros((1, 2))
        self.starting_size = np.zeros((1, 2))

    # ----------------------------------------------------------
    # Métodos de persistencia y configuración
    # ----------------------------------------------------------

    def saveModel(self, context="main"):
        """Guarda el modelo de calibración actual en formato serializado (pickle)."""
        if context in self.clb:
            return pickle.dumps(self.clb[context])

    def loadModel(self, model, context="main"):
        """Carga un modelo de calibración previamente guardado."""
        self.clb[context] = pickle.loads(model)

    def uploadCalibrationMap(self, points, context="main"):
        """Carga una matriz de puntos de calibración personalizada."""
        self.addContext(context)
        self.clb[context].updMatrix(np.array(points))

    # ----------------------------------------------------------
    # Procesamiento de landmarks faciales
    # ----------------------------------------------------------

    def getLandmarks(self, frame):
        """
        Detecta los landmarks faciales y oculares a partir de una imagen (frame).

        Retorna:
            key_points (np.ndarray): Coordenadas procesadas de los landmarks.
            blink (bool): Indica si hay parpadeo.
            subframe (np.ndarray): Región facial recortada.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        self.face.process(frame, self.finder.find(frame))
        face_landmarks = self.face.getLandmarks()
        l_eye = self.face.getLeftEye()
        r_eye = self.face.getRightEye()
        l_eye_landmarks = l_eye.getLandmarks()
        r_eye_landmarks = r_eye.getLandmarks()
        blink = l_eye.getBlink() and r_eye.getBlink()

        # Offset de posición de la cabeza
        x_offset = np.min(face_landmarks[:, 0])
        y_offset = np.min(face_landmarks[:, 1])
        x_width = np.max(face_landmarks[:, 0]) - x_offset
        y_width = np.max(face_landmarks[:, 1]) - y_offset

        head_offset = np.zeros((1, 2))
        scale_x = 1
        scale_y = 1
        if np.array_equal(self.starting_head_position, np.zeros((1, 2))):
            self.starting_head_position = np.array([[x_offset, y_offset]])
            self.starting_size = np.array([[x_width, y_width]])
        else:
            head_offset = np.array([[x_offset, y_offset]]) - self.starting_head_position
            scale_x = self.starting_size[0, 0] / x_width
            scale_y = self.starting_size[0, 1] / y_width

        key_points = np.concatenate(
            (l_eye_landmarks, r_eye_landmarks, np.array([[scale_x, scale_y]]), head_offset)
        )
        key_points[:, 0] -= head_offset[:, 0]
        key_points[:, 1] -= head_offset[:, 1]
        key_points[:, 0] *= scale_x
        key_points[:, 1] *= scale_y
        key_points[-1, 0] = head_offset[:, 0]
        key_points[-1, 1] = head_offset[:, 1]

        subframe = frame[
            int(y_offset):int(y_offset + y_width),
            int(x_offset):int(x_offset + x_width)
        ]
        return key_points, blink, subframe

    def whichAlgorithm(self, context="main"):
        """Devuelve el tipo de algoritmo activo (modelo de calibración actual)."""
        if context in self.clb:
            return self.clb[context].whichAlgorithm()
        return "None"

    def reset(self, context="main"):
        """Reinicia el estado interno del contexto dado."""
        self.filled_points[context] = 0
        if context in self.clb:
            self.addContext(context)

    def setFixation(self, fix):
        """Actualiza el valor de fijación global."""
        self.fix = fix

    # ----------------------------------------------------------
    # Configuración de contexto de calibración
    # ----------------------------------------------------------

    def addContext(self, context):
        """Crea un nuevo contexto de calibración si no existe."""
        if context not in self.clb:
            self.clb[context] = Calibrator_v2(self.calibration_radius)
            self.average_points[context] = np.zeros((20, 2))
            self.filled_points[context] = 0
            self.calibration[context] = False
            self.prev_timestamp[context] = time.time()
            self.prev_point[context] = np.array((0.0, 0.0))
            self.velocity_max[context] = 0
            self.velocity_min[context] = 1e8
            self.fixationTracker[context] = Fixation(0, 0, 100)
            self.key_points_buffer[context] = []

    # ----------------------------------------------------------
    # Ciclo de procesamiento principal
    # ----------------------------------------------------------

    @recoverable(ret_error_params=(None, None))
    def step(self, frame, calibration, width, height, context="main"):
        """
        Realiza un paso de procesamiento del sistema de seguimiento ocular.

        Args:
            frame (np.ndarray): Imagen de entrada.
            calibration (bool): Indica si el sistema está en modo calibración.
            width (int): Ancho de pantalla o display.
            height (int): Altura de pantalla o display.
            context (str): Contexto de ejecución (por defecto "main").

        Returns:
            (Gevent, Cevent): Evento de mirada y evento de calibración.
        """
        self.addContext(context)
        self.calibration[context] = calibration

        key_points, blink, sub_frame = self.getLandmarks(frame)
        self.key_points_buffer[context].append(key_points)
        if len(self.key_points_buffer[context]) > 10:
            self.key_points_buffer[context].pop(0)
        key_points = low_pass_filter_fourier(key_points, 200)

        y_point = self.clb[context].predict(key_points)
        self.average_points[context][1:, :] = self.average_points[context][:-1, :]
        self.average_points[context][0, :] = y_point

        # Acumulador de puntos válidos
        if self.filled_points[context] < self.average_points[context].shape[0] and (
            (y_point != np.array([0.0, 0.0])).any()
        ):
            self.filled_points[context] += 1
        if self.filled_points[context] == 0:
            self.filled_points[context] = 1

        averaged_point = np.sum(
            self.average_points[context][:, :], axis=0
        ) / (self.filled_points[context])

        # Detección de fijación
        fixation = self.fixationTracker[context].process(
            averaged_point[0], averaged_point[1]
        )

        # Cálculo de velocidad (para detectar sacádicos)
        duration = time.time() - self.prev_timestamp[context]
        velocity = np.linalg.norm(averaged_point - self.prev_point[context]) / duration
        self.prev_point[context] = averaged_point
        self.prev_timestamp[context] = time.time()

        self.velocity_max[context] = max(self.velocity_max[context], velocity)
        self.velocity_min[context] = min(self.velocity_min[context], velocity)
        saccades = velocity > (self.velocity_max[context]) / 4

        # Calibración dinámica
        if self.calibration[context] and (
            self.clb[context].insideClbRadius(averaged_point, width, height)
            or self.filled_points[context] < self.average_points[context].shape[0] * 10
        ):
            self.clb[context].add(key_points, self.clb[context].getCurrentPoint(width, height))
        else:
            self.clb[context].post_fit()

        if self.calibration[context] and self.clb[context].insideAcptcRadius(
            averaged_point, width, height
        ):
            if self.clb[context].isReadyToMove():
                self.clb[context].movePoint()

        gevent = Gevent(
            point=averaged_point,
            blink=blink,
            fixation=fixation,
            saccades=saccades,
            sub_frame=sub_frame,
        )
        cevent = Cevent(
            self.clb[context].getCurrentPoint(width, height),
            self.clb[context].acceptance_radius,
            self.clb[context].calibration_radius,
        )
        return (gevent, cevent)
