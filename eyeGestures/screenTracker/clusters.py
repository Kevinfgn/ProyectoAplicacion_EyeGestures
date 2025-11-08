"""
Módulo: clusters.py
Proyecto: Módulo inteligente de calibración ocular basado en EyeGestures
Descripción:
    Este módulo implementa las clases necesarias para agrupar puntos de seguimiento
    ocular (gaze tracking) en regiones de interés (ROI, por sus siglas en inglés) 
    utilizando el algoritmo DBSCAN. 
    
    Cada clúster representa un área donde se concentra la mirada del usuario.
    El propósito principal es detectar y calcular las regiones principales de interés
    dentro de un conjunto de puntos de seguimiento para la calibración y procesamiento
    de pantalla.

Dependencias:
    - numpy
    - scikit-learn (DBSCAN)
"""

from sklearn.cluster import DBSCAN
import numpy as np


class Cluster:
    """
    Clase que representa un clúster de puntos correspondiente a una Región de Interés (ROI).

    Esta clase almacena los puntos pertenecientes al clúster, calcula su centroide, 
    límites (x, y, ancho, alto) y su peso (número de puntos).
    """

    def __init__(self, label, points):
        """
        Constructor de la clase Cluster.

        Args:
            label (int): Identificador del clúster.
            points (array): Lista o arreglo de puntos (x, y) pertenecientes al clúster.
        """
        self.label = label
        self.points = np.array(points)
        self.weight = len(self.points)
        self.__centroid = self.centroid(points)

        x, y, w, h = self.boundaries(points)

        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def centroid(self, points):
        """
        Calcula el centro del clúster (centroide) a partir de sus puntos.

        Combina el centro geométrico del rectángulo delimitador con la media de los puntos
        para obtener una posición más estable.

        Args:
            points (array): Lista de puntos (x, y).

        Returns:
            tuple: Coordenadas (x, y) del centroide.
        """
        x, y, w, h = self.boundaries(points)
        c_x, c_y = (x + w / 2, y + h / 2)
        x, y = sum(points) / len(points)
        c_x, c_y = (c_x + x) / 2, (c_y + y) / 2
        return (c_x, c_y)

    def boundaries(self, points):
        """
        Calcula los límites del clúster en forma de rectángulo delimitador.

        Args:
            points (array): Puntos (x, y) del clúster.

        Returns:
            tuple: Coordenadas (x, y) de la esquina superior izquierda, ancho y alto.
        """
        x = np.min(points[:, 0])
        width = abs(np.max(points[:, 0]) - x)
        y = np.min(points[:, 1])
        height = abs(np.max(points[:, 1]) - y)
        return (x, y, width, height)

    def getBoundaries(self):
        """
        Retorna los límites del clúster.

        Returns:
            tuple: (x, y, ancho, alto)
        """
        return (self.x, self.y, self.w, self.h)

    def getCenter(self):
        """
        Retorna el centroide del clúster (centro de la región de interés).

        Returns:
            tuple: (x, y) centroide.
        """
        return self.__centroid


class Clusters:
    """
    Clase encargada de agrupar puntos rastreados en distintos clústeres
    mediante el algoritmo DBSCAN.

    Esta clase analiza los puntos registrados (por ejemplo, posiciones de mirada)
    para determinar áreas de concentración y seleccionar el clúster principal.
    """

    def __init__(self, buffer):
        """
        Constructor que agrupa los puntos del buffer en clústeres.

        Args:
            buffer (array): Lista o arreglo de puntos (x, y) obtenidos del seguimiento ocular.
        """
        buffer = np.array(buffer)
        self.head = len(buffer) - 1
        self.main_cluster = None
        self.clusters = []

        # Aplicación del algoritmo DBSCAN para agrupamiento espacial
        dbScan = DBSCAN(eps=12, min_samples=3)
        clustering = dbScan.fit(buffer)

        labels = clustering.labels_
        unique_labels = set(labels) - {-1}  # Excluye ruido (-1)

        # Genera los objetos Cluster para cada grupo detectado
        for u_label in unique_labels:
            core_sample_indices = clustering.core_sample_indices_
            cluster_points = buffer[core_sample_indices][
                labels[clustering.core_sample_indices_] == u_label
            ]
            self.clusters.append(Cluster(u_label, cluster_points))

        # Selecciona el clúster principal (el más reciente o más representativo)
        if len(self.clusters) > 0:
            self.main_cluster = self.clusters[labels[self.head]]
        else:
            self.main_cluster = None

    def clearPoints(self):
        """
        [Posible código obsoleto]
        Reinicia el índice de cabeza del buffer de puntos.
        """
        self.head = 0

    def getClusters(self):
        """
        Retorna todos los clústeres detectados.

        Returns:
            list: Lista de objetos Cluster.
        """
        return self.clusters

    def getMainCluster(self):
        """
        Retorna el clúster principal detectado.

        Returns:
            Cluster or None: Clúster principal o None si no se detectó ninguno.
        """
        return self.main_cluster
