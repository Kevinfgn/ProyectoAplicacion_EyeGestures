"""
Módulo: gazeContexter.py
Descripción general:
    Este módulo proporciona las clases y estructuras necesarias para manejar los
    contextos del sistema de seguimiento ocular (gaze tracking). 
    Un "contexto" agrupa todos los datos relevantes de una sesión o instancia de seguimiento,
    incluyendo información del rostro, pupilas, regiones de interés (ROI), fijaciones y calibraciones.

Clases:
    - Contexter: Administra múltiples contextos de seguimiento (creación, actualización, eliminación).
    - Gcontext: Estructura auxiliar que almacena todos los parámetros necesarios de un contexto de seguimiento.
    - GazeContext: Interfaz principal para crear o recuperar contextos de seguimiento a partir de un identificador.

Dependencias:
    - eyeGestures.screenTracker.dataPoints
    - eyeGestures.Fixation
    - eyeGestures.utils.Buffor
"""

import eyeGestures.screenTracker.dataPoints as dp
from eyeGestures.Fixation import Fixation
from eyeGestures.utils import Buffor


class Contexter:
    """
    Clase encargada de administrar los distintos contextos del sistema de seguimiento ocular.
    Permite agregar, eliminar, actualizar y obtener contextos mediante un identificador único.

    Atributos:
        context (dict): Diccionario donde cada clave representa un ID de contexto 
                        y cada valor contiene los datos asociados a dicho contexto.
    """

    def __init__(self):
        """Inicializa la estructura de almacenamiento de contextos."""
        self.context = dict()

    def addContext(self, context_id, object):
        """
        Agrega un nuevo contexto al contenedor si no existe previamente.

        Args:
            context_id (str): Identificador único del contexto.
            object (object): Objeto que contiene los datos del contexto.

        Returns:
            bool: True si se agregó correctamente, False si ya existía.
        """
        if context_id not in self.context.keys():
            self.context[context_id] = object
            return True
        return False

    def rmContext(self, context_id):
        """
        Elimina un contexto existente.

        Args:
            context_id (str): Identificador del contexto a eliminar.

        Returns:
            bool: True si se eliminó correctamente, False si no existía.
        """
        if context_id in self.context.keys():
            del self.context[context_id]
            return True
        return False

    def getContext(self, context_id):
        """
        Devuelve el contexto asociado al identificador indicado.

        Args:
            context_id (str): ID del contexto deseado.

        Returns:
            object | None: El contexto correspondiente o None si no existe.
        """
        if context_id in self.context.keys():
            return self.context[context_id]
        return None

    def updateContext(self, context_id, data):
        """
        Actualiza un contexto existente o lo agrega si no existe.

        Args:
            context_id (str): Identificador del contexto.
            data (object): Nuevo objeto de datos del contexto.

        Returns:
            bool: True siempre, indicando que la actualización o inserción se realizó.
        """
        if context_id in self.context.keys():
            self.context[context_id] = data
            return True
        self.addContext(context_id, data)
        return True

    def getNumberContextes(self):
        """
        Retorna la cantidad total de contextos actualmente almacenados.

        Returns:
            int: Número de contextos activos.
        """
        return len(self.context.keys())


class Gcontext:
    """
    Clase auxiliar que agrupa todos los elementos relevantes de un contexto
    de seguimiento ocular.

    Cada instancia de esta clase contiene referencias a la información 
    necesaria para el procesamiento de la mirada en un entorno determinado.

    Atributos:
        display (object): Información del dispositivo de visualización.
        face (object): Objeto que representa el rostro detectado.
        roi (dp.ScreenROI): Región de interés actual.
        edges (dp.ScreenROI): Bordes del área de calibración o detección.
        cluster_boundaries (dp.ScreenROI): Límites del clúster de puntos.
        gazeBuffor (Buffor): Buffer que almacena el historial de miradas.
        l_pupil, r_pupil (Buffor): Buffers de las posiciones de las pupilas izquierda y derecha.
        l_eye_buff, r_eye_buff (Buffor): Buffers de las posiciones de los ojos.
        fixation (Fixation): Objeto encargado de la detección de fijaciones.
        calibration (bool): Estado de calibración activa o inactiva.
    """

    def __init__(self,
                 display,
                 face,
                 roi,
                 edges,
                 cluster_boundaries,
                 gazeBuffor,
                 l_pupil,
                 r_pupil,
                 l_eye_buff,
                 r_eye_buff,
                 fixation,
                 calibration):

        self.roi = roi
        self.face = face
        self.edges = edges
        self.cluster_boundaries = cluster_boundaries
        self.gazeBuffor = gazeBuffor
        self.l_pupil = l_pupil
        self.r_pupil = r_pupil
        self.l_eye_buff = l_eye_buff
        self.r_eye_buff = r_eye_buff
        self.display = display
        self.fixation = fixation
        self.calibration = calibration


class GazeContext:
    """
    Clase principal que actúa como interfaz de gestión de contextos para el sistema
    de seguimiento ocular. Se encarga de crear, recuperar o actualizar contextos 
    individuales utilizando el `Contexter`.

    Métodos:
        - get(): Crea un nuevo contexto o devuelve uno existente según su ID.
        - update(): Actualiza un contexto ya existente.
    """

    def __init__(self):
        """Inicializa el manejador de contextos de seguimiento."""
        self.contexter = Contexter()

    def get(self,
            id,
            display,
            face=None,
            roi=dp.ScreenROI(285, 105, 80, 15),
            edges=dp.ScreenROI(285, 105, 80, 15),
            cluster_boundaries=dp.ScreenROI(225, 125, 20, 20),
            buffor=Buffor(200),
            l_pupil=Buffor(20),
            r_pupil=Buffor(20),
            l_eye_buff=Buffor(20),
            r_eye_buff=Buffor(20),
            fixation=Fixation(0, 0, 100),
            calibration=False):
        """
        Crea un nuevo contexto de seguimiento ocular o devuelve uno existente
        si el identificador ya fue registrado previamente.

        Args:
            id (str): Identificador único del contexto.
            display (object): Pantalla o dispositivo donde se proyecta el seguimiento.
            face (object, opcional): Objeto que representa el rostro.
            roi, edges, cluster_boundaries (dp.ScreenROI): Regiones geométricas relevantes.
            buffor (Buffor): Buffer para posiciones de mirada.
            l_pupil, r_pupil (Buffor): Buffers de pupilas.
            l_eye_buff, r_eye_buff (Buffor): Buffers de ojos.
            fixation (Fixation): Instancia de detección de fijaciones.
            calibration (bool): Estado de calibración.

        Returns:
            Gcontext: Objeto con toda la información del contexto activo.
        """

        context = Gcontext(display=display,
                           face=face,
                           roi=roi,
                           edges=edges,
                           cluster_boundaries=cluster_boundaries,
                           gazeBuffor=buffor,
                           l_pupil=l_pupil,
                           r_pupil=r_pupil,
                           l_eye_buff=l_eye_buff,
                           r_eye_buff=r_eye_buff,
                           fixation=fixation,
                           calibration=calibration)

        if self.contexter.addContext(id, context):
            return context
        else:
            return self.contexter.getContext(id)

    def update(self, id, context):
        """
        Actualiza un contexto existente en el contexter.

        Args:
            id (str): Identificador del contexto.
            context (Gcontext): Objeto actualizado del contexto.
        """
        self.contexter.updateContext(id, context)
