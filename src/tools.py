"""Las dos herramientas (@tool) del agente.

@tool #1 clasificar_zona   -> envuelve la red neuronal entrenada en la Unidad 4.
@tool #2 evaluar_regla_zona -> reglas de negocio de SafeRest 360 (función determinista).

Las imágenes NUNCA se envían al LLM: la app las registra aquí y el agente solo ve un
identificador corto (img-xxxxxx). La red corre localmente y le devuelve al LLM una sola
línea de texto. Así se ahorran tokens y no salen imágenes hacia la API.
"""

import threading
import uuid

from langchain.tools import tool

from . import reglas

_IMAGENES = {}          # imagen_id -> imagen BGR (np.ndarray)
_ETIQUETAS = {}         # imagen_id -> texto descriptivo (nombre de archivo)
_CANDADO = threading.Lock()
_CLASIFICADOR = None    # instancia de vision.ClasificadorZona, inyectada por la app o el notebook


def configurar_clasificador(clasificador) -> None:
    global _CLASIFICADOR
    _CLASIFICADOR = clasificador


def registrar_imagen(imagen_bgr, etiqueta: str = "") -> str:
    imagen_id = f"img-{uuid.uuid4().hex[:6]}"
    with _CANDADO:
        _IMAGENES[imagen_id] = imagen_bgr
        _ETIQUETAS[imagen_id] = etiqueta
    return imagen_id


def etiqueta_de(imagen_id: str) -> str:
    return _ETIQUETAS.get(imagen_id, "")


@tool
def clasificar_zona(imagen_id: str) -> str:
    """Ejecuta la red neuronal sobre la imagen de una zona del restaurante y dice si está
    ocupada (hay al menos una persona) o vacía, con la probabilidad y la confianza.
    Úsala solo si el mensaje del usuario trae un identificador de imagen como img-3fa2c1.

    Args:
        imagen_id: Identificador de la imagen registrada, por ejemplo img-3fa2c1.
    """
    if _CLASIFICADOR is None:
        return "error=el modelo de la Unidad 4 no está cargado"
    imagen_id = imagen_id.strip()
    with _CANDADO:
        imagen = _IMAGENES.get(imagen_id)
    if imagen is None:
        return f"error=no existe la imagen {imagen_id}"
    r = _CLASIFICADOR.predecir(imagen)
    return (f"imagen={imagen_id} | estado={r['estado']} | p_persona={r['p_persona']:.2f}"
            f" | confianza={r['confianza']:.2f} | modelo={r['modelo']}")


@tool
def evaluar_regla_zona(zona: str, hora: str, estado: str, minutos_en_estado: float = 0, confianza: float = 1.0) -> str:
    """Aplica las reglas de SafeRest 360 y decide si se debe avisar al administrador.
    Zonas: bodega (restringida de 22:00 a 06:00), caja (atendida de 11:00 a 22:00, máximo
    5 min sola) y cocina (atendida de 10:00 a 22:30, máximo 5 min sola).
    Devuelve decision=ALERTA, SIN ALERTA, VERIFICAR (confianza baja) o ERROR.

    Args:
        zona: bodega, caja o cocina.
        hora: Hora del evento en formato de 24 horas HH:MM, por ejemplo 23:30.
        estado: 'ocupada' si hay alguien en la zona o 'vacía' si no hay nadie.
        minutos_en_estado: Minutos que la zona lleva en ese estado; 0 si no se sabe.
        confianza: Confianza que devolvió clasificar_zona; 1.0 si el estado lo informó una persona.
    """
    resultado = reglas.evaluar(zona, hora, estado, minutos_en_estado, confianza)
    return reglas.formatear(resultado)


HERRAMIENTAS = [clasificar_zona, evaluar_regla_zona]
