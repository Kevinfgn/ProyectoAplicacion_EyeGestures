"""
Módulo: calibration_v2.py
Proyecto: Módulo inteligente de calibración ocular basado en EyeGestures
Descripción general:
    Este módulo define una versión avanzada del proceso de calibración,
    basada en aprendizaje automático supervisado (regresión lineal y regularización).
    Utiliza modelos de regresión Ridge y LassoCV para predecir la posición
    visual del usuario con base en muestras recopiladas durante la calibración.

    Se incluye el manejo de concurrencia mediante hilos (threading) para permitir
    la actualización asíncrona de los modelos sin interrumpir la ejecución principal.

Dependencias:
    - numpy: para operaciones vectoriales.
    - sklearn.linear_model: para los modelos de regresión Ridge y LassoCV.
    - sklearn.preprocessing: para normalización de datos.
    - threading: para ejecución paralela de cálculos.
"""

import numpy as np
import sklearn.linear_model as scireg
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import asyncio
import threading


# ============================================================
# FUNCIÓN DE UTILIDAD: DISTANCIA EUCLIDIANA
# ============================================================

def euclidean_distance(point1, point2):
    """
    Calcula la distancia euclidiana entre dos puntos en un espacio n-dimensional.
    
    Args:
        point1 (np.array): Coordenadas del primer punto.
        point2 (np.array): Coordenadas del segundo punto.
    
    Returns:
        float: Distancia euclidiana entre los puntos.
    """
    return np.linalg.norm(point1 - point2)


# ============================================================
# CLASE PRINCIPAL DE CALIBRACIÓN BASADA EN ML
# ============================================================

class Calibrator:
    """
    Clase encargada de gestionar el proceso de calibración utilizando aprendizaje automático.
    Emplea regresión lineal regularizada (Ridge y LassoCV) para mapear las entradas
    de características oculares hacia coordenadas en pantalla.

    Atributos principales:
        X, Y_x, Y_y (list): Datos recopilados para entrenamiento de los modelos.
        reg_x, reg_y (modelo): Modelos de regresión Ridge para los ejes X y Y.
        fitted (bool): Indica si los modelos están entrenados.
        precision_limit (int): Límite mínimo de precisión permitida.
        acceptance_radius (int): Radio de aceptación para considerar un punto calibrado.
        calibration_radius (int): Radio de recolección de puntos de calibración.
        lock (threading.Lock): Mecanismo de exclusión mutua para acceso concurrente seguro.
    """

    PRECISION_LIMIT = 50
    PRECISION_STEP = 10
    ACCEPTANCE_RADIUS = 500

    def __init__(self, CALIBRATION_RADIUS=1000):
        """Inicializa los modelos, variables y estructuras necesarias para la calibración."""
        self.X = []
        self.Y_y = []
        self.Y_x = []
        self.__tmp_X = []
        self.__tmp_Y_y = []
        self.__tmp_Y_x = []
        self.reg = None
        self.reg_x = scireg.Ridge(alpha=0.5)
        self.reg_y = scireg.Ridge(alpha=0.5)
        self.current_algorithm = "Ridge"
        self.fitted = False
        self.cv_not_set = True

        # Matriz que define los puntos de calibración en la pantalla
        self.matrix = CalibrationMatrix()
        
        # Parámetros de precisión y radios de calibración
        self.precision_limit = self.PRECISION_LIMIT
        self.precision_step = self.PRECISION_STEP
        self.acceptance_radius = int(CALIBRATION_RADIUS / 2)
        self.calibration_radius = int(CALIBRATION_RADIUS)

        # Mecanismos de sincronización y ejecución asíncrona
        self.lock = threading.Lock()
        self.calcualtion_coroutine = threading.Thread(target=self.__async_post_fit)
        self.fit_coroutines = [] 

    # ============================================================
    # ENTRENAMIENTO ASÍNCRONO DE LOS MODELOS
    # ============================================================

    def __launch_fit(self):
        """Lanza un hilo para entrenar el modelo sin bloquear la ejecución principal."""
        coroutine = threading.Thread(target=self.__async_fit)
        self.fit_coroutines.append(coroutine)
        coroutine.start()
        self.__join_finished()

    def __join_finished(self):
        """Sincroniza y limpia los hilos de entrenamiento que ya finalizaron."""
        for coroutine in self.fit_coroutines:
            if not coroutine.is_alive():
                coroutine.join()

    # ============================================================
    # RECOLECCIÓN DE PUNTOS DE CALIBRACIÓN
    # ============================================================

    def add(self, x, y):
        """
        Agrega un nuevo punto de calibración (entrada y salida esperada).
        Cada punto incluye un vector de características (x) y una posición real (y).

        Args:
            x (np.array): Vector de características (por ejemplo, posiciones oculares).
            y (np.array): Coordenadas reales del punto en pantalla.
        """
        with self.lock:
            self.__tmp_X.append(x.flatten())
            self.__tmp_Y_y.append(y[1])
            self.__tmp_Y_x.append(y[0])
            self.__launch_fit()

    # ============================================================
    # FUNCIONES INTERNAS DE ENTRENAMIENTO
    # ============================================================

    def __async_fit(self):
        """
        Reentrena los modelos de regresión Ridge de manera asíncrona con los nuevos datos.
        Esta operación es segura en entornos multihilo gracias al uso de `self.lock`.
        """
        try:
            with self.lock:
                __fit_tmp_X = np.array(self.__tmp_X + self.X, dtype=object)
                __fit_tmp_Y_y = np.array(self.__tmp_Y_y + self.Y_y)
                __fit_tmp_Y_x = np.array(self.__tmp_Y_x + self.Y_x)
                self.reg_x.fit(__fit_tmp_X, __fit_tmp_Y_x)
                self.reg_y.fit(__fit_tmp_X, __fit_tmp_Y_y)
                self.fitted = True
        except Exception as e:
            print(f"Exception as {e}")

    def __async_post_fit(self):
        """
        Calcula modelos de ajuste más avanzados (LassoCV) para optimizar la precisión
        después del entrenamiento inicial con Ridge. Este proceso se ejecuta en segundo plano.
        """
        try:
            tmp_fixations_x = scireg.LassoCV(cv=50, max_iter=10000)
            tmp_fixations_y = scireg.LassoCV(cv=50, max_iter=10000)

            __tmp_X = np.array(self.X)
            __tmp_Y_y = np.array(self.Y_y)
            __tmp_Y_x = np.array(self.Y_x)

            tmp_fixations_x.fit(__tmp_X, __tmp_Y_x)
            tmp_fixations_y.fit(__tmp_X, __tmp_Y_y)

            with self.lock:
                self.fixations_x = tmp_fixations_x
                self.fixations_y = tmp_fixations_y
                self.fitted = True
                self.current_algorithm = "LassoCV"
        except Exception as e:
            print(f"Exception as {e}")
            self.cv_not_set = True
        pass

    # ============================================================
    # MÉTODOS DE CONTROL Y ESTADO
    # ============================================================

    def post_fit(self):
        """Inicia la ejecución del ajuste avanzado si aún no se ha realizado."""
        if self.cv_not_set:
            self.cv_not_set = False

    def whichAlgorithm(self):
        """Devuelve el nombre del algoritmo actualmente en uso (Ridge o LassoCV)."""
        with self.lock:
            return self.current_algorithm

    # ============================================================
    # PREDICCIÓN DE COORDENADAS
    # ============================================================

    def predict(self, x):
        """
        Predice las coordenadas en pantalla basadas en el vector de entrada (características oculares).

        Args:
            x (np.array): Vector de características del punto a evaluar.

        Returns:
            np.array: Coordenadas predichas [x, y] o [0, 0] si el modelo aún no está entrenado.
        """
        with self.lock:
            if self.fitted:
                x = x.flatten().reshape(1, -1)
                y_x = self.reg_x.predict(x)[0]
                y_y = self.reg_y.predict(x)[0]
                return np.array([y_x, y_y])
            else:
                return np.array([0.0, 0.0])

    # ============================================================
    # MANEJO DE PUNTOS Y PRECISIÓN
    # ============================================================

    def movePoint(self):
        """
        Mueve el punto de calibración actual, consolidando los datos temporales 
        en el conjunto principal y actualizando la matriz de calibración.
        """
        with self.lock:
            self.X = self.X + self.__tmp_X
            self.Y_y = self.Y_y + self.__tmp_Y_y
            self.Y_x = self.Y_x + self.__tmp_Y_x
            self.matrix.movePoint()
            self.__tmp_X = []
            self.__tmp_Y_y = []
            self.__tmp_Y_x = []

    def isReadyToMove(self):
        """
        Determina si se han recolectado suficientes datos para mover al siguiente punto.
        Returns True cuando hay al menos 30 muestras.
        """
        return len(self.__tmp_X) > 30  # valor empírico para estabilidad

    # ============================================================
    # OPERACIONES CON MATRIZ DE CALIBRACIÓN
    # ============================================================

    def getCurrentPoint(self, width, heigth):
        """Obtiene el punto actual de calibración en coordenadas de pantalla."""
        return self.matrix.getCurrentPoint(width, heigth)

    def updMatrix(self, points):
        """Actualiza la matriz de puntos de calibración."""
        return self.matrix.updMatrix(points)

    # ============================================================
    # RESTABLECER ESTADO DE CALIBRACIÓN
    # ============================================================

    def unfit(self):
        """Reinicia los parámetros de calibración y radios de aceptación."""
        self.acceptance_radius = self.ACCEPTANCE_RADIUS
        self.calibration_radius = self.CALIBRATION_RADIUS
        self.fitted = False

    def increase_precision(self):
        """
        Ajusta gradualmente los radios de aceptación y calibración para 
        mejorar la precisión del sistema después de cada paso exitoso.
        """
        if self.acceptance_radius > self.precision_limit:
            self.acceptance_radius -= self.precision_step
        if self.calibration_radius > self.precision_limit and self.acceptance_radius < self.calibration_radius:
            self.calibration_radius -= self.precision_step

    # ============================================================
    # VALIDACIÓN DE PUNTOS DENTRO DE RADIOS
    # ============================================================

    def insideClbRadius(self, point, width, height):
        """
        Verifica si un punto se encuentra dentro del radio de calibración.
        """
        return euclidean_distance(point, self.getCurrentPoint(width, height)) < self.calibration_radius
    
    def insideAcptcRadius(self, point, width, height):
        """
        Verifica si un punto se encuentra dentro del radio de aceptación.
        """
        return euclidean_distance(point, self.getCurrentPoint(width, height)) < self.acceptance_radius


# ============================================================
# CLASE DE MATRIZ DE CALIBRACIÓN
# ============================================================

class CalibrationMatrix:
    """
    Clase que representa una matriz de puntos de calibración distribuidos sobre la pantalla.
    Define el orden y posiciones relativas que guían el proceso de calibración.

    Atributos:
        points (np.array): Coordenadas normalizadas (0.0 - 1.0) para cada punto de calibración.
        iterator (int): Índice del punto actual.
    """

    def __init__(self):
        """Inicializa la matriz con una distribución predefinida de 25 puntos."""
        self.iterator = 0
        self.points = np.array([
            [1,0.5],[0.75,0.5],[0.5,0.5],[0.25,0.5],[0.0,0.5],
            [1.0,1.0],[0.75,1.0],[0.5,1.0],[0.25,1.0],[0.0,1.0],
            [1.0,0.0],[0.75,0.0],[0.5,0.0],[0.25,0.0],[0.0,0.0],
            [0.0,0.75],[0.75,0.75],[0.5,0.75],[0.25,0.75],[0.0,0.75],
            [1.0,0.25],[0.75,0.25],[0.5,0.25],[0.25,0.25],[0.0,0.25]
        ])

    def updMatrix(self, points):
        """Actualiza los puntos de calibración y reinicia el iterador."""
        self.points = points
        self.iterator = 0

    def movePoint(self):
        """Avanza al siguiente punto en la matriz de calibración."""
        self.iterator += 1
        self.iterator %= len(self.points)

    def getCurrentPoint(self, width=1.0, height=1.0):
        """
        Devuelve la posición actual de calibración ajustada al tamaño real de la pantalla.
        """
        it = self.iterator
        return np.array([self.points[it,0] * width, self.points[it,1] * height])
