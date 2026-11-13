"""Configuración compartida por los notebooks, las herramientas del agente y la app."""

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
DIR_MODELOS = RAIZ / "models"
DIR_ASSETS = RAIZ / "assets"
DIR_FALLOS = DIR_ASSETS / "fallos"
DIR_EJEMPLOS = DIR_ASSETS / "ejemplos"
RUTA_INFO_MODELO = DIR_MODELOS / "modelo_info.json"

# Imagen de entrada de la red (alto = ancho, en píxeles)
TAMANO_IMAGEN = 160

# Salida de la red: probabilidad de que haya al menos una persona en la imagen
CLASE_POSITIVA = "ocupada"   # hay al menos una persona
CLASE_NEGATIVA = "vacía"     # no hay nadie
UMBRAL_DECISION = 0.5

# Modelo de Google AI Studio usado en la Unidad 5 del curso
MODELO_GEMINI = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
