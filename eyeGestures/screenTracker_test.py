"""
Módulo: screen_tracker_test.py
Descripción general:
    Este módulo contiene un conjunto de pruebas unitarias destinadas a validar
    el comportamiento de la función `screen2display()` implementada en la clase
    `ScreenProcessor` dentro del módulo `screenTracker`.

    Dicha función realiza la conversión de coordenadas desde un sistema de
    referencia "virtual" (pantalla o ROI de seguimiento) hacia un sistema
    físico o "real" (pantalla del monitor o display).

    Estas pruebas aseguran que:
        - Los puntos dentro de la ROI se escalen correctamente.
        - Los valores fuera del rango se limiten a los bordes del display.
        - Las coordenadas de entrada y salida se comporten conforme a las
          proporciones de anchura y altura especificadas.

Dependencias:
    - eyeGestures.screenTracker.dataPoints
    - eyeGestures.screenTracker.screenTracker
"""

import eyeGestures.screenTracker.dataPoints as dp
import eyeGestures.screenTracker.screenTracker as scrtr

# =====================================================
# Parámetros comunes para las pruebas
# =====================================================

# Tamaño de la pantalla virtual utilizada por ScreenProcessor
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

# Región de interés (ROI) dentro de la pantalla virtual
ROI_WIDTH = 50
ROI_HEIGHT = 50

# Dimensiones del display real
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 1800


# =====================================================
# Pruebas de conversión de coordenadas ScreenProcessor
# =====================================================

def test_screen2display_1_1():
    """
    Verifica que un punto cercano al origen (1,1) dentro de la ROI
    sea escalado correctamente en función de las dimensiones del display.
    """
    point = (1, 1)
    roi = dp.ScreenROI(0, 0, ROI_WIDTH, ROI_HEIGHT)
    display = dp.Display(DISPLAY_WIDTH, DISPLAY_HEIGHT, 0, 0)
    tracker = scrtr.ScreenProcessor()

    p = tracker.screen2display(point, roi, display)

    assert p == (point[0] / ROI_WIDTH * DISPLAY_WIDTH,
                 point[1] / ROI_HEIGHT * DISPLAY_HEIGHT)


def test_screen2display_55_1():
    """
    Verifica que un punto con coordenada X fuera del rango de la ROI
    (x=55 > ancho=50) sea limitado al borde máximo del display.
    """
    point = (55, 1)
    roi = dp.ScreenROI(0, 0, ROI_WIDTH, ROI_HEIGHT)
    display = dp.Display(DISPLAY_WIDTH, DISPLAY_HEIGHT, 0, 0)
    tracker = scrtr.ScreenProcessor()

    p = tracker.screen2display(point, roi, display)

    assert p == (DISPLAY_WIDTH,
                 point[1] / ROI_HEIGHT * DISPLAY_HEIGHT)


def test_screen2display_55_55():
    """
    Verifica que un punto fuera del rango en ambos ejes (x,y)
    sea limitado correctamente al borde inferior derecho del display.
    """
    point = (55, 55)
    roi = dp.ScreenROI(0, 0, ROI_WIDTH, ROI_HEIGHT)
    display = dp.Display(DISPLAY_WIDTH, DISPLAY_HEIGHT, 0, 0)
    tracker = scrtr.ScreenProcessor()

    p = tracker.screen2display(point, roi, display)

    assert p == (DISPLAY_WIDTH, DISPLAY_HEIGHT)


def test_screen2display_0_0():
    """
    Verifica que un punto en el origen (0,0) se escale correctamente
    al origen del display sin desplazamientos.
    """
    point = (0, 0)
    roi = dp.ScreenROI(0, 0, ROI_WIDTH, ROI_HEIGHT)
    display = dp.Display(DISPLAY_WIDTH, DISPLAY_HEIGHT, 0, 0)
    tracker = scrtr.ScreenProcessor()

    p = tracker.screen2display(point, roi, display)
    assert p == (0, 0)


def test_screen2display_1_1_pos_50_50():
    """
    Verifica la conversión cuando la ROI está desplazada en (50,50).
    El punto dentro de esa región debe escalarse correctamente
    considerando el desplazamiento de origen.
    """
    point = (51, 51)
    pos = (50, 50)

    roi = dp.ScreenROI(pos[0], pos[1], ROI_WIDTH, ROI_HEIGHT)
    display = dp.Display(DISPLAY_WIDTH, DISPLAY_HEIGHT, 0, 0)
    tracker = scrtr.ScreenProcessor()

    p = tracker.screen2display(point, roi, display)

    assert p == ((point[0] - pos[0]) / ROI_WIDTH * DISPLAY_WIDTH,
                 (point[1] - pos[1]) / ROI_HEIGHT * DISPLAY_HEIGHT)


def test_screen2display_55_55_pos_50_50():
    """
    Verifica que un punto fuera de los límites de una ROI desplazada
    sea restringido correctamente al borde máximo del display.
    """
    point = (105, 105)
    pos = (50, 50)

    roi = dp.ScreenROI(pos[0], pos[1], ROI_WIDTH, ROI_HEIGHT)
    display = dp.Display(DISPLAY_WIDTH, DISPLAY_HEIGHT, 0, 0)
    tracker = scrtr.ScreenProcessor()

    p = tracker.screen2display(point, roi, display)

    assert p == (DISPLAY_WIDTH, DISPLAY_HEIGHT)


def test_screen2display_0_0_pos_50_50():
    """
    Verifica que un punto anterior al origen de una ROI desplazada
    (coordenadas negativas respecto al área de interés)
    sea recortado al origen del display (0,0).
    """
    point = (0, 0)
    pos = (50, 50)

    roi = dp.ScreenROI(pos[0], pos[1], ROI_WIDTH, ROI_HEIGHT)
    display = dp.Display(DISPLAY_WIDTH, DISPLAY_HEIGHT, 0, 0)
    tracker = scrtr.ScreenProcessor()

    p = tracker.screen2display(point, roi, display)
    assert p == (0, 0)
