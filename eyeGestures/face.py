"""
Módulo: face.py
Descripción general:
    Este módulo se encarga de la detección y extracción del rostro dentro de una imagen,
    así como del manejo de los ojos detectados en el mismo mediante el uso de MediaPipe FaceMesh.
    Proporciona dos clases principales:
        - FaceFinder: para localizar rostros en imágenes.
        - Face: para representar un rostro y manejar sus componentes (ojos y landmarks).

Dependencias:
    - cv2: procesamiento de imágenes.
    - numpy: cálculos matriciales y manipulación de datos.
    - mediapipe: detección de malla facial (FaceMesh).
    - eyeGestures.eye: clase que gestiona los ojos y sus características.
"""

import cv2
import numpy as np
import mediapipe as mp
import eyeGestures.eye as eye


# ============================================================
# CLASE: FaceFinder
# ============================================================

class FaceFinder:
    """
    Clase encargada de detectar el rostro en una imagen utilizando MediaPipe FaceMesh.
    Proporciona una interfaz simple para extraer los puntos de referencia (landmarks)
    del rostro detectado.

    Atributos:
        mp_face_mesh: instancia del modelo de MediaPipe FaceMesh.
    """

    def __init__(self):
        """
        Inicializa el detector de rostros (FaceMesh) de MediaPipe.
        Se configura con refinamiento de landmarks y confianza mínima de detección/tracking.
        """
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def find(self, image):
        """
        Detecta un rostro en la imagen dada y retorna la estructura de landmarks.

        Args:
            image (np.ndarray): Imagen en formato BGR (como la devuelve OpenCV).

        Returns:
            face_mesh (object | None): Objeto con los landmarks faciales detectados, 
            o None si no se encuentra un rostro.
        """
        assert (len(image.shape) > 2), "La imagen debe tener tres canales (color)."

        try:
            # Convierte la imagen a RGB, requerido por MediaPipe
            face_mesh = self.mp_face_mesh.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )

            # Si no se detecta ningún rostro
            if face_mesh.multi_face_landmarks is None:
                return None

            return face_mesh

        except Exception as e:
            print(f"Exception in FaceFinder: {e}")
            return None


# ============================================================
# CLASE: Face
# ============================================================

class Face:
    """
    Clase que representa un rostro detectado en una imagen.
    Contiene la información de los landmarks faciales, los ojos izquierdo y derecho,
    y permite obtener características geométricas del rostro.

    Atributos:
        eyeLeft (Eye): Instancia del ojo izquierdo.
        eyeRight (Eye): Instancia del ojo derecho.
        landmarks (np.ndarray): Coordenadas de los puntos detectados del rostro.
    """

    def __init__(self):
        """
        Inicializa un objeto Face con sus dos ojos (izquierdo y derecho) y sin landmarks aún.
        """
        self.eyeLeft = eye.Eye(0)
        self.eyeRight = eye.Eye(1)
        self.landmarks = None

    # ============================================================
    # MÉTODOS DE ACCESO
    # ============================================================

    def getBoundingBox(self):
        """
        Calcula el rectángulo delimitador (bounding box) del rostro completo
        con base en los landmarks faciales detectados.

        Returns:
            tuple: (x, y, width, height) del rostro. Si no hay landmarks, retorna (0, 0, 0, 0).
        """
        if self.landmarks is not None:
            margin = 0
            min_x = np.min(self.landmarks[:, 0]) - margin
            max_x = np.max(self.landmarks[:, 0]) + margin
            min_y = np.min(self.landmarks[:, 1]) - margin
            max_y = np.max(self.landmarks[:, 1]) + margin

            width = int((max_x - min_x))
            height = int((max_y - min_y))
            x = int(min_x)
            y = int(min_y)
            return (x, y, width, height)

        # En caso de que no haya datos
        return (0, 0, 0, 0)

    def getLeftEye(self):
        """Devuelve la instancia del ojo izquierdo."""
        return self.eyeLeft

    def getRightEye(self):
        """Devuelve la instancia del ojo derecho."""
        return self.eyeRight

    def getLandmarks(self):
        """Devuelve los landmarks faciales detectados."""
        return self.landmarks

    # ============================================================
    # PROCESAMIENTO DE LANDMARKS
    # ============================================================

    def _landmarks(self, face):
        """
        Convierte los puntos de landmark de MediaPipe a coordenadas absolutas (pixeles).

        Args:
            face (object): Objeto devuelto por MediaPipe con los landmarks faciales.

        Returns:
            np.ndarray: Array de coordenadas (x, y) del rostro.
        """
        __complex_landmark_points = face.multi_face_landmarks
        __complex_landmarks = __complex_landmark_points[0].landmark

        __face_landmarks = []
        for landmark in __complex_landmarks:
            __face_landmarks.append((
                landmark.x * self.image_w,
                landmark.y * self.image_h
            ))

        return np.array(__face_landmarks)

    def process(self, image, face):
        """
        Procesa una imagen para extraer y actualizar la información del rostro,
        incluyendo landmarks y actualización de los ojos.

        Args:
            image (np.ndarray): Imagen que contiene el rostro.
            face (object): Objeto de MediaPipe con los landmarks faciales detectados.
        """
        self.face = face
        self.image_h, self.image_w, _ = image.shape

        # Obtiene los landmarks normalizados a píxeles
        self.landmarks = self._landmarks(self.face)

        # Determina el offset con base en el bounding box del rostro
        x, y, _, _ = self.getBoundingBox()
        offset = np.array((x, y))

        # Actualiza los ojos izquierdo y derecho con los nuevos landmarks
        self.eyeLeft.update(image, self.landmarks, offset)
        self.eyeRight.update(image, self.landmarks, offset)
