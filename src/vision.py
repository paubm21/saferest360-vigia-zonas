"""Preprocesamiento con OpenCV e inferencia de la red neuronal (Unidad 4).

El MISMO preprocesamiento se usa al entrenar (notebook) y al predecir (herramienta del
agente y app), para que la red reciba en producción exactamente lo que vio al entrenar.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .config import (
    CLASE_NEGATIVA,
    CLASE_POSITIVA,
    DIR_MODELOS,
    RUTA_INFO_MODELO,
    TAMANO_IMAGEN,
    UMBRAL_DECISION,
)

COLOR_RELLENO = (114, 114, 114)  # gris neutro para el relleno del letterbox


def leer_imagen_bgr(fuente) -> np.ndarray:
    """Lee una imagen desde una ruta o desde bytes y la devuelve en BGR uint8 (formato OpenCV)."""
    if isinstance(fuente, (bytes, bytearray)):
        buffer = np.frombuffer(fuente, dtype=np.uint8)
        imagen = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    else:
        imagen = cv2.imread(str(fuente), cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError(f"OpenCV no pudo leer la imagen: {fuente if not isinstance(fuente, bytes) else 'bytes'}")
    return imagen


def preprocesar(imagen_bgr: np.ndarray, tamano: int = TAMANO_IMAGEN) -> np.ndarray:
    """BGR -> RGB y redimensión con letterbox (sin recortar ni deformar).

    Se rellena en lugar de recortar porque un recorte central puede dejar por fuera a una
    persona que está en el borde de la imagen, y la etiqueta diría "ocupada" sin que la red
    vea a nadie. Devuelve un arreglo uint8 de forma (tamano, tamano, 3); la normalización
    de valores la hace una capa Rescaling dentro del propio modelo.
    """
    rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    alto, ancho = rgb.shape[:2]
    escala = tamano / max(alto, ancho)
    nuevo_alto = max(1, round(alto * escala))
    nuevo_ancho = max(1, round(ancho * escala))
    redimensionada = cv2.resize(rgb, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
    arriba = (tamano - nuevo_alto) // 2
    abajo = tamano - nuevo_alto - arriba
    izquierda = (tamano - nuevo_ancho) // 2
    derecha = tamano - nuevo_ancho - izquierda
    return cv2.copyMakeBorder(
        redimensionada, arriba, abajo, izquierda, derecha, cv2.BORDER_CONSTANT, value=COLOR_RELLENO
    )


def brillo_medio(imagen_bgr: np.ndarray) -> float:
    """Brillo medio (canal V de HSV, 0-255). Sirve para analizar el desempeño con poca luz."""
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def interpretar_probabilidad(p_persona: float, umbral: float = UMBRAL_DECISION) -> dict:
    """Convierte la salida sigmoide de la red en estado + confianza."""
    estado = CLASE_POSITIVA if p_persona >= umbral else CLASE_NEGATIVA
    confianza = p_persona if estado == CLASE_POSITIVA else 1.0 - p_persona
    return {"estado": estado, "p_persona": round(float(p_persona), 4), "confianza": round(float(confianza), 4)}


class ClasificadorZona:
    """Carga una sola vez el modelo .keras exportado por el notebook de entrenamiento."""

    def __init__(self, ruta_modelo=None):
        from tensorflow import keras  # import diferido: el resto del módulo no necesita TensorFlow

        self.info = {}
        if RUTA_INFO_MODELO.exists():
            self.info = json.loads(RUTA_INFO_MODELO.read_text(encoding="utf-8"))
        ruta = Path(ruta_modelo) if ruta_modelo else DIR_MODELOS / self.info.get("archivo_modelo", "")
        if not ruta.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo en {ruta}. Ejecuten primero notebooks/entrenamiento_u4.ipynb."
            )
        self.modelo = keras.models.load_model(ruta)
        self.umbral = float(self.info.get("umbral", UMBRAL_DECISION))
        self.nombre = self.info.get("configuracion", ruta.stem)

    def predecir(self, imagen_bgr: np.ndarray) -> dict:
        x = preprocesar(imagen_bgr)[np.newaxis].astype("float32")
        p_persona = float(self.modelo.predict(x, verbose=0)[0][0])
        resultado = interpretar_probabilidad(p_persona, self.umbral)
        resultado["modelo"] = self.nombre
        return resultado
