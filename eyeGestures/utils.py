"""
Módulo: utils.py
Descripción general:
    Este módulo contiene un conjunto de utilidades generales utilizadas en el sistema
    de seguimiento ocular (eye-tracking), incluyendo temporización, manejo de errores,
    filtrado de señales, buffers de datos y captura de video sin búfer.

    Las funciones y clases aquí definidas proporcionan herramientas de soporte
    para los módulos de procesamiento de imagen, calibración y análisis de mirada.

Incluye:
    - Decoradores de utilidad (`recoverable`, `timeit`)
    - Filtro pasa-bajo por transformada de Fourier
    - Generador de cuadrícula de imágenes (`make_image_grid`)
    - Clases auxiliares (`var`, `Buffor`, `VideoCapture`)
"""

import time
import queue
import pickle
import platform
import threading
import cv2
import numpy as np


# =====================================================
# Decoradores y utilidades generales
# =====================================================

def recoverable(ret_error_params=()):
    """
    Decorador que permite ejecutar una función de forma segura
    y recuperar en caso de error, devolviendo un valor por defecto.

    Args:
        ret_error_params (tuple): Valor que se devolverá si ocurre una excepción.

    Returns:
        function: Función decorada que maneja excepciones de forma controlada.
    """
    def decorator(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error detectado en {func.__name__}: {e}")
                return ret_error_params
        return inner
    return decorator


def timeit(func):
    """
    Decorador que mide el tiempo de ejecución de una función
    e imprime el resultado en consola.

    Uso:
        @timeit
        def funcion():
            ...
    """
    def inner(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        print(f"Tiempo transcurrido: {time.time() - start:.4f} s")
        return ret
    return inner


# =====================================================
# Procesamiento de señales y datos
# =====================================================

def low_pass_filter_fourier(data, cutoff_frequency):
    """
    Aplica un filtro pasa-bajo basado en Transformada de Fourier (FFT)
    de forma columna por columna a un conjunto de datos.

    Args:
        data (np.ndarray): Matriz de datos (N x M) donde cada columna es una señal.
        cutoff_frequency (float): Frecuencia de corte para el filtro.

    Returns:
        np.ndarray: Datos filtrados de la misma forma que la entrada.
    """
    filtered_data = np.zeros_like(data, dtype=float)
    for col in range(data.shape[1]):
        fft_data = np.fft.fft(data[:, col])
        frequencies = np.fft.fftfreq(len(data[:, col]))
        fft_data[np.abs(frequencies) > cutoff_frequency] = 0
        filtered_data[:, col] = np.fft.ifft(fft_data).real
    return filtered_data


def shape_to_np(shape, dtype="int"):
    """
    Convierte un objeto de forma facial (ej. dlib shape predictor)
    en un arreglo NumPy con coordenadas (x, y).

    Args:
        shape: Objeto con puntos faciales.
        dtype (str): Tipo de dato de salida.

    Returns:
        np.ndarray: Matriz (68x2) con coordenadas de los landmarks.
    """
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def make_image_grid(images, rows, cols):
    """
    Crea una cuadrícula de imágenes a partir de una lista.

    Args:
        images (list): Lista de imágenes (todas deben tener el mismo tamaño y tipo).
        rows (int): Número de filas en la cuadrícula.
        cols (int): Número de columnas en la cuadrícula.

    Returns:
        np.ndarray: Imagen compuesta en forma de cuadrícula.
    """
    assert images, "La lista de imágenes está vacía"

    img_h, img_w = images[0].shape[:2]
    if len(images[0].shape) > 2:
        grid_image = np.zeros((img_h * rows, img_w * cols, images[0].shape[2]), dtype=np.uint8)
    else:
        grid_image = np.zeros((img_h * rows, img_w * cols), dtype=np.uint8)

    for i, img in enumerate(images):
        if i >= rows * cols:
            break
        row = i // cols
        col = i % cols
        grid_image[row * img_h:(row + 1) * img_h, col * img_w:(col + 1) * img_w] = img

    return grid_image


# =====================================================
# Clases auxiliares
# =====================================================

class var:
    """
    Clase contenedora simple para almacenar y actualizar un valor mutable.
    """

    def __init__(self, var):
        self.__var = var

    def set(self, var):
        """Actualiza el valor almacenado."""
        self.__var = var

    def get(self):
        """Obtiene el valor actual almacenado."""
        return self.__var


class Buffor:
    """
    Clase para almacenar una secuencia de datos con un tamaño fijo (FIFO).

    Permite almacenar una ventana deslizante de valores para aplicar
    operaciones como promedios o limpieza del búfer.
    """

    def __init__(self, length):
        self.length = length
        self.__buffor = []

    def add(self, var):
        """Agrega un valor al búfer, eliminando el más antiguo si está lleno."""
        if len(self.__buffor) >= self.length:
            self.__buffor.pop(0)
        self.__buffor.append(var)

    def getAvg(self, lenght=0):
        """Calcula el promedio de los últimos elementos del búfer."""
        return np.sum(self.__buffor[-lenght:], axis=0) / len(self.__buffor[-lenght:])

    def getBuffor(self):
        """Devuelve el contenido completo del búfer."""
        return self.__buffor

    def loadBuffor(self, buffor):
        """Carga manualmente un conjunto de datos en el búfer."""
        self.__buffor = buffor

    def getLast(self):
        """Devuelve el primer elemento almacenado (más antiguo)."""
        return self.__buffor[0]

    def getFirst(self):
        """Devuelve el último elemento almacenado (más reciente)."""
        return self.__buffor[len(self.__buffor) - 1]

    def getLen(self):
        """Devuelve el número actual de elementos almacenados."""
        return len(self.__buffor)

    def isFull(self):
        """Indica si el búfer ha alcanzado su capacidad máxima."""
        return len(self.__buffor) >= self.length

    def flush(self):
        """Conserva el último valor y limpia el resto del búfer."""
        tmp = self.__buffor[-1]
        self.__buffor = [tmp]

    def clear(self):
        """Vacía completamente el búfer."""
        self.__buffor = []


# =====================================================
# Captura de video sin búfer
# =====================================================

class VideoCapture:
    """
    Clase que envuelve la captura de video de OpenCV,
    implementando una lectura sin búfer (bufforless).

    Esto permite acceder siempre al último frame disponible
    sin retraso por acumulación de cuadros antiguos.
    """

    def __init__(self, name, bufforless=True):
        self.bufforless = bufforless
        self.run = True

        # Detección de tipo de entrada (archivo o cámara)
        if isinstance(name, str):
            self.stream = not (".pkl" in name)
        else:
            self.stream = True

        # Fuente en vivo (cámara)
        if self.stream:
            self.prev_frame = None
            self.__openCam(name)
            self.q = queue.Queue()
            self.t = threading.Thread(target=self.__reader)
            self.t.start()
        # Fuente desde archivo serializado (.pkl)
        else:
            self.frames = []
            with open(name, 'rb') as file:
                self.frames = pickle.load(file)

    def __openCam(self, name):
        """Abre la cámara, detectando el backend adecuado según el sistema operativo."""
        if isinstance(name, int):
            if "Windows" in platform.system():
                self.cap = cv2.VideoCapture(name, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(name)
            if self.cap is None or not self.cap.isOpened():
                print(f"No se pudo abrir la cámara {name}. Intentando con {name + 1}...")
                if name + 1 < 10:
                    self.__openCam(name + 1)
        else:
            self.cap = cv2.VideoCapture(name)

    def __reader(self):
        """Hilo de lectura continua de frames."""
        while self.run:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty() and self.bufforless:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((ret, frame))
        self.flush()

    def flush(self):
        """Limpia la cola de frames acumulados."""
        while not self.q.empty():
            self.q.get()

    def read(self):
        """
        Devuelve el frame más reciente disponible.

        Returns:
            tuple(bool, np.ndarray): Estado y frame actual.
        """
        if self.stream:
            return self.q.get()
        else:
            frame = self.frames.pop(0)
            self.frames.pop(0)
            return ((len(self.frames) >= 1), frame)

    def close(self):
        """Detiene el hilo de captura y libera la cámara."""
        self.run = False
        self.t.join()
        self.cap.release()
