"""
Módulo: screenTracker.py
Proyecto: Módulo inteligente de calibración ocular basado en EyeGestures
Descripción:
    Este módulo contiene las clases y funciones necesarias para realizar el 
    seguimiento de los puntos oculares en pantalla, la actualización de la región 
    de interés (ROI) y la calibración dinámica del sistema EyeGestures.

    Se encarga de la conversión entre coordenadas de pantalla y de display, 
    detección de bordes, escalado adaptativo de regiones y agrupación de puntos
    mediante análisis de calor y clustering.

Dependencias:
    - math
    - numpy
    - scipy.signal
    - sklearn.cluster.DBSCAN
    - eyeGestures.screenTracker.dataPoints (dp)
    - eyeGestures.screenTracker.clusters (Clusters)
    - eyeGestures.screenTracker.heatmap (Heatmap)
"""

import math
import numpy as np
from scipy import signal
from sklearn.cluster import DBSCAN

import eyeGestures.screenTracker.dataPoints as dp
from eyeGestures.screenTracker.clusters import Clusters
from eyeGestures.screenTracker.heatmap import Heatmap


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def detect_if_inside(point, rect):
    """
    Determina si un punto se encuentra dentro de un rectángulo (ROI).

    Args:
        point (tuple): Coordenadas (x, y) del punto a evaluar.
        rect (dp.ScreenROI): Objeto que representa la región rectangular (ROI).

    Returns:
        bool: True si el punto está dentro de la ROI, False en caso contrario.
    """
    px = point[0]
    py = point[1]
    x, y, width, height = rect.getBoundaries()

    x_in = x < px and px < x + width
    y_in = y < py and py < y + height

    return x_in and y_in


def detect_edges(roi, display, point_on_screen, point_on_display):
    """
    Detecta los bordes del ROI en función de la posición del punto y los límites del display.

    Args:
        roi (dp.ScreenROI): Región de interés actual.
        display (dp.Display): Objeto que representa el área visible del sistema.
        point_on_screen (tuple): Coordenadas del punto en la pantalla.
        point_on_display (tuple): Coordenadas del punto en el display.

    Returns:
        dp.ScreenROI: Nueva ROI ajustada a los límites detectados.
    """
    (s_x, s_y) = point_on_screen
    (d_x, d_y) = point_on_display

    x, y, width, height = roi.getBoundaries()
    new_roi = dp.ScreenROI(x, y, width, height)

    # Ajuste horizontal
    if d_x <= 0:
        new_roi.width = new_roi.width + abs(new_roi.x - s_x)
        new_roi.x = s_x

    if d_x >= display.width:
        new_roi.width = abs(new_roi.x - s_x)

    # Ajuste vertical
    if d_y <= 0:
        new_roi.height = new_roi.height + abs(new_roi.y - s_y)
        new_roi.y = s_y

    if d_y >= display.height:
        new_roi.height = abs(new_roi.y - s_y)

    return new_roi


def rescale_h(roi, scale_h, change=0.5):
    """
    Cambia la altura del ROI según un factor de escala.

    Args:
        roi (dp.ScreenROI): Región de interés.
        scale_h (float): Factor de escala sobre la altura.
        change (float): Umbral mínimo de cambio permitido.

    Returns:
        float: Nueva altura del ROI.
    """
    scale_diff_h = abs(1.0 - scale_h)
    ret_heigth = roi.height

    if scale_diff_h > change:
        ret_heigth = roi.height / 1.0 * scale_h

    return ret_heigth


def rescale_w(roi, scale_w, change=0.5):
    """
    Cambia el ancho del ROI según un factor de escala.

    Args:
        roi (dp.ScreenROI): Región de interés.
        scale_w (float): Factor de escala sobre el ancho.
        change (float): Umbral mínimo de cambio permitido.

    Returns:
        float: Nuevo ancho del ROI.
    """
    scale_diff_w = abs(1.0 - scale_w)
    ret_width = roi.width

    if scale_diff_w > change:
        ret_width = roi.width / 1.0 * scale_w

    return ret_width


def scaleDown(roi, edge, scale):
    """
    Escala hacia abajo el ROI hasta ajustarlo a los bordes detectados.

    Args:
        roi (dp.ScreenROI): Región de interés actual.
        edge (dp.ScreenROI): Límites detectados mediante clustering.
        scale (float): Factor de escala (negativo para reducción).

    Returns:
        dp.ScreenROI: Nueva ROI ajustada.
    """
    (_, _, cluster_w, cluster_h) = edge.getBoundaries()

    new_roi = dp.ScreenROI(roi.x, roi.y, roi.width, roi.height)

    if cluster_w < roi.width:
        new_scale_w = 1.0 + scale
        new_roi.width = rescale_w(roi, new_scale_w, scale)

    if cluster_h < roi.height:
        new_scale_h = 1.0 + scale
        new_roi.height = rescale_h(roi, new_scale_h, scale)

    return new_roi


def scaleUp(roi, roi2, scale):
    """
    Escala hacia arriba el ROI hasta ajustarlo a otro ROI de referencia.

    Args:
        roi (dp.ScreenROI): Región actual.
        roi2 (dp.ScreenROI): Región objetivo.
        scale (float): Factor de escala positivo.

    Returns:
        dp.ScreenROI: Nueva ROI expandida.
    """
    (_, _, roi2_w, roi2_h) = roi2.getBoundaries()
    new_roi = dp.ScreenROI(roi.x, roi.y, roi.width, roi.height)

    if roi2_w > roi.width:
        new_roi.width = rescale_w(roi, 1.0 + scale, scale)

    if roi2_h > roi.height:
        new_roi.height = rescale_h(roi, 1.0 + scale, scale)

    return new_roi


# ============================================================
# CLASES PRINCIPALES
# ============================================================

class ScreenProcessor:
    """
    Clase encargada del procesamiento y conversión de coordenadas entre pantalla y display.

    Implementa la lógica de calibración automática, detección de bordes y
    actualización de la región de interés (ROI) según los datos del heatmap y clustering.
    """

    def __init__(self):
        pass

    def process(self, point, point_offset, buffor_length, roi, edges, screen, display, heatmap):
        """
        Función principal que procesa un punto y actualiza la información de ROI.

        Args:
            point (tuple): Punto actual rastreado.
            point_offset (tuple): Desplazamiento del punto.
            buffor_length (int): Longitud del buffer de puntos.
            roi, edges, screen, display, heatmap: Objetos de referencia del sistema.

        Returns:
            tuple: (punto_en_display, porcentaje_de_cercanía)
        """
        s_point_offset = self.display2screen(point_offset, screen, display)
        p_on_display = self.screen2display(
            [point[0] + s_point_offset[0], point[1] + s_point_offset[1]], roi, display)

        if buffor_length > 20:
            new_edges = detect_edges(roi, display, point, p_on_display)
            edges.x, edges.y = new_edges.x, new_edges.y
            edges.width, edges.height = new_edges.width, new_edges.height

        p_on_display = (p_on_display[0] + display.offset_x, p_on_display[1] + display.offset_y)

        (_, _, roi_w, roi_h) = heatmap.getBoundaries()
        closeness_percentage = (roi_w * roi_h) / (screen.width * screen.height)
        return (p_on_display, closeness_percentage)

    def update(self, roi, edges, cluster, heatmap):
        """
        Actualiza la región de interés (ROI) según los nuevos datos de cluster y heatmap.

        Args:
            roi (dp.ScreenROI): Región actual.
            edges (dp.ScreenROI): Bordes detectados.
            cluster (Cluster): Agrupación de puntos.
            heatmap (Heatmap): Mapa de calor actual.

        Returns:
            dp.ScreenROI: Nueva región de interés actualizada.
        """
        (x, y) = heatmap.getCenter()
        new_roi = dp.ScreenROI(roi.x, roi.y, roi.width, roi.height)

        new_roi.setCenter(x, y)
        edges.setCenter(x, y)

        # Escalado hacia arriba o abajo dependiendo de los límites detectados
        new_roi = scaleUp(new_roi, edges, scale=0.1)
        new_roi = scaleDown(new_roi, cluster, scale=-0.1)
        return new_roi

    def screen2display(self, screen_point, screen, display):
        """
        Convierte un punto de coordenadas de pantalla a coordenadas de display.
        """
        s_x, s_y = screen_point[0], screen_point[1]
        d_x = int((s_x - screen.x) / screen.width * display.width)
        d_y = int((s_y - screen.y) / screen.height * display.height)

        d_x = max(min(d_x, display.width), 0)
        d_y = max(min(d_y, display.height), 0)
        return (d_x, d_y)

    def display2screen(self, display_point, screen, display):
        """
        Convierte un punto de coordenadas de display a coordenadas de pantalla.
        """
        d_x, d_y = display_point[0], display_point[1]
        s_x = int((d_x) / display.width * screen.width)
        s_y = int((d_y) / display.height * screen.height)
        return (s_x, s_y)


class ScreenManager:
    """
    Clase encargada de coordinar el procesamiento principal del sistema de seguimiento ocular.

    Gestiona el procesamiento, la calibración y la actualización continua de la ROI.
    """

    def __init__(self):
        self.screen_processor = ScreenProcessor()

    def process(self, buffor, roi, edges, screen, display, calibration, offset):
        """
        Ejecuta el procesamiento completo del sistema, integrando heatmap, clustering y calibración.

        Args:
            buffor: Buffer con puntos rastreados.
            roi: Región de interés actual.
            edges: Bordes de la ROI.
            screen, display: Parámetros físicos del entorno.
            calibration (bool): Indica si está activo el modo calibración.
            offset (tuple): Desplazamiento del punto base.

        Returns:
            tuple: (punto, roi_actualizada, cluster_principal)
        """
        heatmap = Heatmap(screen.width, screen.height, buffor.getBuffor())
        cluster = Clusters(buffor.getBuffor()).getMainCluster()

        if cluster is not None:
            if calibration:
                roi = self.screen_processor.update(roi, edges, cluster, heatmap)

            p, percentage = self.screen_processor.process(
                buffor.getAvg(20),
                (offset[0], offset[1]),
                len(buffor.getBuffor()),
                roi,
                edges,
                screen,
                display,
                heatmap
            )

            return (p, roi, cluster)

        return ([0, 0], roi, cluster)
