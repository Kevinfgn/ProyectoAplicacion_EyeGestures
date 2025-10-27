"""
Módulo: eye.py
Descripción general:
    Este módulo proporciona una clase para la extracción y procesamiento de la región del ojo
    dentro de una imagen facial. Utiliza los puntos de referencia (landmarks) detectados 
    mediante MediaPipe FaceMesh para segmentar el ojo, calcular su centro, detectar parpadeos,
    estimar la dirección de la mirada y obtener la imagen recortada de la región ocular.

Dependencias:
    - cv2: para operaciones de procesamiento de imagen.
    - numpy: para cálculos vectoriales y matriciales.
    - mediapipe: para obtener los índices de los puntos del mapeo facial.
"""

import cv2
import numpy as np
import mediapipe as mp


class Eye:
    """
    Clase que representa un ojo detectado en un rostro.
    Gestiona la extracción de características como el parpadeo, apertura, dirección
    de la mirada y coordenadas del centro del ojo a partir de los landmarks faciales.

    Atributos:
        LEFT_EYE_KEYPOINTS (list): Índices de puntos faciales del ojo izquierdo.
        RIGHT_EYE_KEYPOINTS (list): Índices de puntos faciales del ojo derecho.
        LEFT_EYE_PUPIL_KEYPOINT (list): Índice del punto correspondiente a la pupila izquierda.
        RIGHT_EYE_PUPIL_KEYPOINT (list): Índice del punto correspondiente a la pupila derecha.
        scale (tuple): Tamaño de referencia para la escala del ojo.
    """

    LEFT_EYE_KEYPOINTS = np.array(list(mp.solutions.face_mesh.FACEMESH_LEFT_EYE))[:, 0]
    RIGHT_EYE_KEYPOINTS = np.array(list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE))[:, 0]
    LEFT_EYE_IRIS_KEYPOINT = []
    RIGHT_EYE_IRIS_KEYPOINT = []
    LEFT_EYE_PUPIL_KEYPOINT = [473]
    RIGHT_EYE_PUPIL_KEYPOINT = [468]

    scale = (150, 100)

    def __init__(self, side: int):
        """
        Inicializa un objeto Eye especificando si es el ojo izquierdo o derecho.

        Args:
            side (int): 0 para ojo izquierdo, 1 para ojo derecho.
        """
        if side == 1:
            self.side = "right"
            self.pupil_index = self.RIGHT_EYE_PUPIL_KEYPOINT
        elif side == 0:
            self.side = "left"
            self.pupil_index = self.LEFT_EYE_PUPIL_KEYPOINT

        # Propiedades del ojo
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.center_x = 0
        self.center_y = 0
        self.image = None
        self.pupil = None
        self.offset = None
        self.region = None
        self.cut_image = None
        self.landmarks = None

    # ============================================================
    # MÉTODOS PRINCIPALES
    # ============================================================

    def update(self, image: np.ndarray, landmarks: list, offset: np.ndarray):
        """
        Actualiza la información del ojo a partir de una nueva imagen y puntos faciales.

        Args:
            image (np.ndarray): Imagen original del rostro.
            landmarks (list): Lista de coordenadas de landmarks faciales.
            offset (np.ndarray): Vector de desplazamiento aplicado a las coordenadas.
        """
        self.image = image
        self.offset = offset
        self.landmarks = landmarks

        # Determina los puntos correspondientes al ojo según el lado
        if self.side == "right":
            self.region = np.array(landmarks[self.RIGHT_EYE_KEYPOINTS])
        elif self.side == "left":
            self.region = np.array(landmarks[self.LEFT_EYE_KEYPOINTS])

        # Coordenada de la pupila estimada por MediaPipe
        self.pupil = landmarks[self.pupil_index][0]

        # Procesa y extrae región del ojo
        self._process(self.image, self.region)

    # ============================================================
    # FUNCIONES DE ACCESO Y CÁLCULO
    # ============================================================

    def getCenter(self):
        """Devuelve las coordenadas del centro del ojo."""
        return (self.center_x, self.center_y)

    def getPos(self):
        """Devuelve la posición (x, y) de la esquina superior izquierda del ojo."""
        return (self.x, self.y)

    def getPupil(self):
        """Devuelve la posición estimada de la pupila."""
        return self.pupil

    def getBlink(self):
        """
        Detecta si el ojo está cerrado (parpadeo).
        Returns:
            bool: True si el ojo parece cerrado.
        """
        return (self.height) <= 3  # margen empírico

    def getImage(self):
        """Devuelve la imagen recortada del ojo a partir del rostro."""
        return self.cut_image

    def getGaze(self, gaze_buffor, y_correction=0, x_correction=0):
        """
        Calcula la dirección estimada de la mirada con base en el desplazamiento
        de la pupila respecto al centro del ojo y los puntos periféricos del mismo.

        Args:
            gaze_buffor: Objeto que almacena el promedio de vectores de mirada.
            y_correction (float): Corrección vertical manual.
            x_correction (float): Corrección horizontal manual.

        Returns:
            np.array: Vector promedio de dirección de la mirada.
        """
        center = np.array((self.center_x, self.center_y)) - self.offset
        region_corrected = self.region - self.offset
        pupil_corrected = self.pupil - self.offset

        vectors = region_corrected - center
        pupil = pupil_corrected - center
        vectors = vectors - pupil

        gaze_vector = np.zeros((2))
        gaze_vector[1] = np.sum(vectors, axis=0)[1] * 10 - y_correction
        gaze_vector[0] = -np.sum(vectors, axis=0)[0] * 10 - x_correction

        gaze_buffor.add(gaze_vector)
        return gaze_buffor.getAvg()

    def getOpenness(self):
        """Calcula el grado de apertura del ojo (basado en la altura detectada)."""
        return self.height / 2

    def getLandmarks(self):
        """Devuelve los landmarks (puntos faciales) asociados al ojo."""
        return self.region
    
    def getBoundingBox(self):
        """Devuelve el rectángulo delimitador del ojo (x, y, ancho, alto)."""
        return (self.x, self.y, self.width, self.height)

    # ============================================================
    # PROCESAMIENTO DE LA REGIÓN OCULAR
    # ============================================================

    def _process(self, image, region):
        """
        Procesa la imagen del rostro para extraer la región del ojo.
        Crea una máscara binaria que aísla el área ocular y calcula sus dimensiones.

        Args:
            image (np.ndarray): Imagen del rostro.
            region (np.ndarray): Conjunto de puntos que delimitan el ojo.
        """
        h, w, _ = image.shape

        # Máscara del ojo
        mask = np.full((h, w), 0, dtype=np.uint8)
        background = np.zeros((h, w), dtype=np.uint8)
        region_int = np.array(region, dtype=np.int32)

        # Se rellena la región del ojo
        cv2.fillPoly(mask, [region_int], 0)

        # Se aplica máscara a la imagen original
        masked_image = cv2.bitwise_not(background, cv2.cvtColor(
            image.copy(), cv2.COLOR_BGR2GRAY), mask=mask)

        # Se calculan los límites del área ocular
        margin = 2
        min_x = np.min(region_int[:, 0]) - margin
        max_x = np.max(region_int[:, 0]) + margin
        min_y = np.min(region_int[:, 1]) - margin
        max_y = np.max(region_int[:, 1]) + margin

        # Propiedades geométricas
        self.x = min_x
        self.y = min_y
        self.width = np.max(region_int[:, 0]) - np.min(region_int[:, 0])
        self.height = np.max(region_int[:, 1]) - np.min(region_int[:, 1])
        self.center_x = (min_x + max_x) / 2
        self.center_y = (min_y + max_y) / 2

        # Ajuste correctivo (MediaPipe a veces reporta mal el eje Y de la pupila)
        self.pupil[1] = np.min(region[:, 1])

        # Se recorta la región del ojo
        self.cut_image = masked_image[min_y:max_y, min_x:max_x]
