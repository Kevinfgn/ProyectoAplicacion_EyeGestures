"""
Módulo: scalling_test.py
Descripción general:
    Este módulo contiene pruebas unitarias automatizadas (usando pytest)
    para validar las funciones de escalamiento de regiones de interés (ROI)
    utilizadas en el módulo `screenTracker`.

    Las pruebas comprueban el comportamiento de las funciones:
    - `scaleUp`: Ampliación proporcional de la ROI.
    - `scaleDown`: Reducción proporcional de la ROI.

    Se evalúan tanto casos de comportamiento esperado (aumento/disminución del tamaño)
    como valores exactos cuando se aplica un factor de escala conocido.

Dependencias:
    - pytest
    - eyeGestures.screenTracker.screenTracker
    - eyeGestures.screenTracker.dataPoints
"""

import pytest
import eyeGestures.screenTracker.screenTracker as scrtr
import eyeGestures.screenTracker.dataPoints as dp


# =====================================================
# Parámetros comunes de las pruebas
# =====================================================

ROI_WIDTH = 30   # Ancho de la región de interés base
ROI_HEIGHT = 20  # Altura de la región de interés base


# =====================================================
# Pruebas de escalamiento (funciones scaleUp / scaleDown)
# =====================================================

def test_scaleUpTest():
    """
    Verifica que la función `scaleUp` incremente el tamaño
    (ancho y alto) de una ROI en comparación con su tamaño original.
    """
    pos = (50, 50)
    BIGGER = 10

    roi = dp.ScreenROI(pos[0], pos[1], ROI_WIDTH, ROI_HEIGHT)
    roi2 = dp.ScreenROI(pos[0], pos[1], ROI_WIDTH + BIGGER, ROI_HEIGHT + BIGGER)

    bigger_roi = scrtr.scaleUp(roi, roi2, 0.1)

    assert bigger_roi.width > roi.width
    assert bigger_roi.height > roi.height


def test_scaleUpTest_known():
    """
    Verifica que `scaleUp` incremente el tamaño exacto de la ROI
    en un 10% cuando se usa un factor de escala conocido (0.1).
    """
    pos = (50, 50)
    BIGGER = 10

    roi = dp.ScreenROI(pos[0], pos[1], ROI_WIDTH, ROI_HEIGHT)
    roi2 = dp.ScreenROI(pos[0], pos[1], ROI_WIDTH + BIGGER, ROI_HEIGHT + BIGGER)

    bigger_roi = scrtr.scaleUp(roi, roi2, 0.1)

    assert int(bigger_roi.width) == int(ROI_WIDTH * 1.1)
    assert int(bigger_roi.height) == int(ROI_HEIGHT * 1.1)


def test_scaleDownTest():
    """
    Verifica que la función `scaleDown` reduzca el tamaño
    (ancho y alto) de la ROI con respecto a su valor original.
    """
    pos = (50, 50)
    SMALLER = 10

    roi = dp.ScreenROI(pos[0], pos[1], ROI_WIDTH, ROI_HEIGHT)
    roi2 = dp.ScreenROI(pos[0], pos[1], ROI_WIDTH - SMALLER, ROI_HEIGHT - SMALLER)

    smaller_roi = scrtr.scaleDown(roi, roi2, -0.1)

    assert smaller_roi.width < roi.width
    assert smaller_roi.height < roi.height


def test_scaleDownTest_known():
    """
    Verifica que `scaleDown` reduzca el tamaño exacto de la ROI
    en un 10% cuando se aplica un factor de escala negativo (-0.1).
    """
    pos = (50, 50)
    SMALLER = 10

    roi = dp.ScreenROI(pos[0], pos[1], ROI_WIDTH, ROI_HEIGHT)
    roi2 = dp.ScreenROI(pos[0], pos[1], ROI_WIDTH - SMALLER, ROI_HEIGHT - SMALLER)

    smaller_roi = scrtr.scaleDown(roi, roi2, -0.1)

    assert int(smaller_roi.width) == int(ROI_WIDTH * 0.9)
    assert int(smaller_roi.height) == int(ROI_HEIGHT * 0.9)
