"""
Módulo: gazeEstimator.py
Descripción general:
    Este módulo implementa la lógica central para el seguimiento ocular (gaze tracking),
    desde la detección de rostro y ojos hasta la estimación de la dirección de la mirada.
    Combina el uso de la biblioteca MediaPipe para la detección facial y de ojos,
    junto con procesadores de pupila y filtros de seguimiento que permiten
    determinar la posición de la mirada en pantalla.

    Es uno de los módulos principales del sistema EyeGestures, encargado de generar
    un objeto de evento ("Gevent") que representa el estado actual de la mirada,
    parpadeos, fijación y regiones de interés.

Clases:
    - GazeTracker: Clase principal que gestiona el procesamiento de imágenes,
      detección facial, cálculo de mirada y actualización de contextos de seguimiento.

Funciones:
    - isInside(): Verifica si un punto se encuentra dentro de un círculo, útil para
      detectar movimientos dentro de un radio de fijación.
"""

import numpy as np
from eyeGestures.gevent import Gevent
from eyeGestures.face import FaceFinder, Face
from eyeGestures.Fixation import Fixation
from eyeGestures.processing import EyeProcessor
from eyeGestures.gazeContexter import GazeContext
from eyeGestures.screenTracker.screenTracker import ScreenManager
import eyeGestures.screenTracker.dataPoints as dp
from eyeGestures.utils import Buffor


def isInside(circle_x, circle_y, r, x, y):
    """
    Función que verifica si un punto (x, y) se encuentra dentro del área
    de un círculo centrado en (circle_x, circle_y) con radio r.

    Args:
        circle_x (float): Coordenada X del centro del círculo.
        circle_y (float): Coordenada Y del centro del círculo.
        r (float): Radio del círculo.
        x (float): Coordenada X del punto.
        y (float): Coordenada Y del punto.

    Returns:
        bool: True si el punto está dentro del círculo, False si está fuera.
    """
    if (x - circle_x) * (x - circle_x) + (y - circle_y) * (y - circle_y) <= r * r:
        return True
    else:
        return False


class GazeTracker:
    """
    Clase principal encargada del procesamiento de imágenes para determinar
    la posición estimada de la mirada en pantalla.

    Combina diferentes componentes:
        - Detección de rostro y ojos mediante MediaPipe.
        - Procesamiento de pupila mediante EyeProcessor.
        - Seguimiento del rostro con FaceFinder y Face.
        - Estimación de la dirección de la mirada con ScreenManager.

    Atributos principales:
        screen (dp.Screen): Representación del área visible o pantalla.
        face (Face): Objeto que contiene la información facial detectada.
        finder (FaceFinder): Detector de rostros de MediaPipe.
        screen_man (ScreenManager): Gestor del mapeo entre la pantalla y el área de mirada.
        GContext (GazeContext): Administrador de contextos de seguimiento ocular.
    """

    N_FEATURES = 16  # Número de características a considerar por defecto.

    def __init__(
        self,
        screen_width,
        screen_heigth,
        eye_screen_w,
        eye_screen_h,
        roi_x,
        roi_y,
        roi_width,
        roi_height,
        monitor_offset_x=0,
        monitor_offset_y=0,
    ):
        """Inicializa los parámetros principales del seguimiento ocular."""
        self.screen = dp.Screen(screen_width, screen_heigth)

        self.offset_x = 0
        self.offset_y = 0

        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height

        self.eye_screen_w = eye_screen_w
        self.eye_screen_h = eye_screen_h

        # Procesadores independientes para cada ojo
        self.eyeProcessorLeft = EyeProcessor(eye_screen_w, eye_screen_h)
        self.eyeProcessorRight = EyeProcessor(eye_screen_w, eye_screen_h)

        # Administrador de pantalla y contextos
        self.screen_man = ScreenManager()
        self.finder = FaceFinder()
        self.face = Face()

        # Variables auxiliares
        self.__headDir = [0.5, 0.5]
        self.point_screen = [0.0, 0.0]
        self.freezed_point = [0.0, 0.0]
        self.GContext = GazeContext()

    # ------------------------------------------------------------------------------------------
    # MÉTODOS INTERNOS
    # ------------------------------------------------------------------------------------------

    def __gaze_intersection(self, l_eye, r_eye, l_buff, r_buff):
        """
        Calcula el punto de intersección de las líneas de mirada de ambos ojos.
        Se obtiene a partir de las coordenadas de la pupila y la dirección del vector de mirada.

        Args:
            l_eye, r_eye: Objetos Eye representando cada ojo.
            l_buff, r_buff: Buffers de mirada asociados.

        Returns:
            tuple: Coordenadas (x, y) de la intersección estimada.
        """
        l_pupil = l_eye.getPupil()
        l_gaze = l_eye.getGaze(l_buff)

        r_pupil = r_eye.getPupil()
        r_gaze = r_eye.getGaze(r_buff)

        l_end = l_gaze + l_pupil
        r_end = r_gaze + r_pupil

        l_m = (l_end[1] - l_pupil[1]) / (l_end[0] - l_pupil[0])
        r_m = (r_end[1] - r_pupil[1]) / (r_end[0] - r_pupil[0])

        l_b = l_end[1] - l_m * l_end[0]
        r_b = r_end[1] - r_m * r_end[0]

        i_x = (r_b - l_b) / (l_m - r_m)
        i_y = r_m * i_x + r_b
        return (i_x, i_y)

    def __pupil(self, eye, eyeProcessor, intersection_x, buffor):
        """
        Procesa la posición promedio de la pupila para un ojo,
        actualizando el buffer correspondiente.

        Args:
            eye (Eye): Objeto del ojo (izquierdo o derecho).
            eyeProcessor (EyeProcessor): Procesador asignado al ojo.
            intersection_x (float): Coordenada X de intersección.
            buffor (Buffor): Buffer donde se almacenan los puntos de pupila.

        Returns:
            tuple: Punto estimado (x, y) y buffer actualizado.
        """
        eyeProcessor.append(eye.getPupil(), eye.getLandmarks(), buffor)
        point = eyeProcessor.getAvgPupil(self.eye_screen_w, self.eye_screen_h, buffor)
        point = np.array((int(intersection_x), point[1]))
        return point, buffor

    # ------------------------------------------------------------------------------------------
    # MÉTODOS PÚBLICOS
    # ------------------------------------------------------------------------------------------

    def estimate(
        self,
        image,
        display,
        context_id,
        calibration,
        fixation_freeze=0.7,
        freeze_radius=20,
        offset_x=0,
        offset_y=0,
    ):
        """
        Procesa una imagen del rostro para estimar la posición de la mirada.

        Args:
            image (ndarray): Imagen de entrada con el rostro del usuario.
            display (object): Pantalla o dispositivo donde se proyecta la mirada.
            context_id (str): Identificador del contexto actual.
            calibration (bool): Indica si el sistema está en modo calibración.
            fixation_freeze (float): Umbral de fijación (0 a 1).
            freeze_radius (int): Radio de estabilidad de la mirada.
            offset_x, offset_y (float): Desplazamientos en píxeles.

        Returns:
            Gevent | None: Evento de mirada con información procesada o None si no se detecta rostro.
        """
        event = None
        face_mesh = self.getFeatures(image)
        if not face_mesh:
            return None

        if face_mesh.multi_face_landmarks:
            self.face.process(image, face_mesh)

        # Crear o recuperar contexto
        context = self.GContext.get(
            context_id,
            display,
            face=None,
            roi=dp.ScreenROI(self.roi_x, self.roi_y, self.roi_width, self.roi_height),
            edges=dp.ScreenROI(285, 105, 80, 15),
            cluster_boundaries=dp.ScreenROI(225, 125, 20, 20),
            buffor=Buffor(200),
            l_pupil=Buffor(20),
            r_pupil=Buffor(20),
            l_eye_buff=Buffor(20),
            r_eye_buff=Buffor(20),
            fixation=Fixation(0, 0, 100),
        )
        context.calibration = calibration

        # Procesamiento del rostro
        if not self.face is None:
            if self.face.landmarks is None:
                return event

            if context.face == None:
                x, y, w, h = self.face.getBoundingBox()
                i_w = self.face.image_w
                i_h = self.face.image_h
                context.face = (x, y, w, h, i_w, i_h)

            l_eye = self.face.getLeftEye()
            r_eye = self.face.getRightEye()

            # Cálculo del punto de intersección de las miradas
            intersection_x, _ = self.__gaze_intersection(
                l_eye, r_eye, context.l_eye_buff, context.r_eye_buff
            )
            l_point, l_buffor = self.__pupil(
                l_eye, self.eyeProcessorLeft, intersection_x, context.l_pupil
            )
            r_point, r_buffor = self.__pupil(
                r_eye, self.eyeProcessorRight, intersection_x, context.r_pupil
            )

            context.l_pupil = l_buffor
            context.r_pupil = r_buffor

            compound_point = np.array(((l_point + r_point) / 2), dtype=np.uint32)

            blink = l_eye.getBlink() or r_eye.getBlink()
            if blink != True:
                context.gazeBuffor.add(compound_point)

            # Ajuste de tamaño de ROI en función del cambio de distancia del rostro
            if blink != True:
                face_x, face_y, face_w, face_h = self.face.getBoundingBox()
                image_w = self.face.image_w
                image_h = self.face.image_h

                face_w_perc = face_w / image_w
                face_h_perc = face_h / image_h

                c_face_x, c_face_y, c_face_w, c_face_h, c_image_w, c_image_h = (
                    context.face
                )

                c_face_w_perc = c_face_w / c_image_w
                c_face_h_perc = c_face_h / c_image_h

                if abs(face_w_perc / c_face_w_perc - 1.0) > 0.02:
                    context.roi.width = context.roi.width * abs(
                        face_w_perc / c_face_w_perc
                    )
                    context.gazeBuffor.flush()
                if abs(face_h_perc / c_face_h_perc - 1.0) > 0.02:
                    context.roi.height = context.roi.height * abs(
                        face_h_perc / c_face_h_perc
                    )
                    context.gazeBuffor.flush()

            # Proceso de conversión entre ROI y pantalla
            self.point_screen, roi, cluster = self.screen_man.process(
                context.gazeBuffor,
                context.roi,
                context.edges,
                self.screen,
                context.display,
                context.calibration,
                (offset_x, offset_y),
            )

            context.roi = roi
            if cluster:
                x, y, width, height = cluster.getBoundaries()
                context.cluster_boundaries.x = x
                context.cluster_boundaries.y = y
                context.cluster_boundaries.width = width
                context.cluster_boundaries.height = height

            self.GContext.update(context_id, context)

            # Procesamiento de fijaciones
            fix = context.fixation.process(self.point_screen[0], self.point_screen[1])

            # Si se supera el umbral de fijación, se congela el punto de mirada
            if fix > fixation_freeze:
                r = freeze_radius
                if not isInside(
                    self.freezed_point[0],
                    self.freezed_point[1],
                    r,
                    self.point_screen[0],
                    self.point_screen[1],
                ):
                    self.freezed_point = self.point_screen

                event = Gevent(
                    self.freezed_point,
                    blink,
                    fix,
                    l_eye,
                    r_eye,
                    display,
                    context.roi,
                    context.edges,
                    context.cluster_boundaries,
                    context_id,
                )
            else:
                self.freezed_point = self.point_screen
                event = Gevent(
                    self.point_screen,
                    blink,
                    fix,
                    l_eye,
                    r_eye,
                    display,
                    context.roi,
                    context.edges,
                    context.cluster_boundaries,
                    context_id,
                )

        return event

    def getFeatures(self, image):
        """
        Devuelve los puntos de referencia (landmarks) detectados del rostro.

        Args:
            image (ndarray): Imagen en formato BGR.

        Returns:
            face_mesh (object): Resultado del análisis de MediaPipe con landmarks faciales.
        """
        face_mesh = self.finder.find(image)
        return face_mesh
