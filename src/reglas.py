"""Reglas de negocio de SafeRest 360 (herramienta #2 del agente).

Son las reglas por zona y horario del brief del proyecto: una zona restringida no debe
tener a nadie en su horario restringido, y una zona atendida no debe quedarse sola más
del tiempo permitido (mínimo 2 minutos). Son deterministas: no usan el LLM.
Si cambian ZONAS, actualicen también el docstring de evaluar_regla_zona en tools.py.
"""

import re
import unicodedata

MIN_MINUTOS_ABANDONO = 2      # el brief exige un tiempo mínimo de confirmación
CONFIANZA_MINIMA = 0.70       # por debajo, la decisión pasa a un humano

ZONAS = {
    "bodega": {
        "tipo": "restringida",
        "inicio": "22:00",
        "fin": "06:00",
        "descripcion": "bodega de insumos; nadie debe estar en ella en horario restringido",
    },
    "caja": {
        "tipo": "atendida",
        "inicio": "11:00",
        "fin": "22:00",
        "max_min_sin_persona": 5,
        "descripcion": "caja registradora; debe estar atendida en horario de servicio",
    },
    "cocina": {
        "tipo": "atendida",
        "inicio": "10:00",
        "fin": "22:30",
        "max_min_sin_persona": 5,
        "descripcion": "área de preparación; debe estar atendida en horario de servicio",
    },
}

ACCION_AVISO = (
    "Enviar aviso por WhatsApp al administrador para que revise. "
    "El aviso no es una acusación: la decisión final la toma una persona."
)


def _sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def normalizar_zona(zona: str) -> str:
    return _sin_tildes(str(zona)).strip().lower()


def normalizar_estado(estado: str) -> str:
    e = _sin_tildes(str(estado)).strip().lower()
    if e in {"ocupada", "ocupado", "con persona", "con personas", "persona", "hay alguien"}:
        return "ocupada"
    if e in {"vacia", "vacio", "sin persona", "sin personas", "sola", "solo", "nadie"}:
        return "vacía"
    raise ValueError(f"Estado no reconocido: '{estado}'. Use 'ocupada' o 'vacía'.")


def a_minutos(hora: str) -> int:
    """Convierte '23:30', '11:30 pm', '11:30 p. m.' o '3 pm' a minutos desde la medianoche."""
    texto = _sin_tildes(str(hora)).strip().lower().replace(".", "").replace(" ", "")
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)?", texto)
    if not m:
        raise ValueError(f"Hora no reconocida: '{hora}'. Use el formato HH:MM de 24 horas.")
    h, mins, sufijo = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if sufijo:
        if not 1 <= h <= 12:
            raise ValueError(f"Hora no válida: '{hora}'.")
        h = h % 12 + (12 if sufijo == "pm" else 0)
    if not (0 <= h <= 23 and 0 <= mins <= 59):
        raise ValueError(f"Hora no válida: '{hora}'.")
    return h * 60 + mins


def en_ventana(hora: str, inicio: str, fin: str) -> bool:
    """True si la hora cae en [inicio, fin). Soporta ventanas que cruzan la medianoche."""
    t, a, b = a_minutos(hora), a_minutos(inicio), a_minutos(fin)
    return a <= t < b if a < b else (t >= a or t < b)


def evaluar(zona: str, hora: str, estado: str, minutos_en_estado: float = 0, confianza: float = 1.0) -> dict:
    """Decide si se debe avisar. Devuelve un diccionario con decision ALERTA / SIN ALERTA / VERIFICAR / ERROR."""
    clave = normalizar_zona(zona)
    if clave not in ZONAS:
        return {"decision": "ERROR", "motivo": f"Zona '{zona}' no configurada. Zonas: {', '.join(ZONAS)}."}
    try:
        estado_n = normalizar_estado(estado)
        dentro = en_ventana(hora, ZONAS[clave]["inicio"], ZONAS[clave]["fin"])
    except ValueError as error:
        return {"decision": "ERROR", "motivo": str(error)}

    cfg = ZONAS[clave]
    base = {"zona": clave, "hora": hora, "estado": estado_n}
    ventana = f"{cfg['inicio']}-{cfg['fin']}"

    if float(confianza) < CONFIANZA_MINIMA:
        return {**base, "decision": "VERIFICAR",
                "motivo": f"La red no está segura (confianza {float(confianza):.2f} < {CONFIANZA_MINIMA:.2f}).",
                "accion": "Un humano debe revisar la imagen antes de decidir si se avisa."}

    if cfg["tipo"] == "restringida":
        if not dentro:
            return {**base, "decision": "SIN ALERTA", "motivo": f"Fuera del horario restringido ({ventana})."}
        if estado_n == "ocupada":
            return {**base, "decision": "ALERTA", "motivo": f"Persona en zona restringida dentro del horario {ventana}.",
                    "accion": ACCION_AVISO}
        return {**base, "decision": "SIN ALERTA", "motivo": "Zona restringida sin personas."}

    # Zona atendida: aviso si queda sola más del tiempo permitido en horario de servicio
    limite = max(MIN_MINUTOS_ABANDONO, cfg["max_min_sin_persona"])
    if not dentro:
        return {**base, "decision": "SIN ALERTA", "motivo": f"Fuera del horario de servicio ({ventana})."}
    if estado_n == "ocupada":
        return {**base, "decision": "SIN ALERTA", "motivo": "La zona está atendida."}
    if float(minutos_en_estado) >= limite:
        return {**base, "decision": "ALERTA",
                "motivo": f"Zona sola {float(minutos_en_estado):g} min; el máximo es {limite} min.",
                "accion": ACCION_AVISO}
    faltan = limite - float(minutos_en_estado)
    return {**base, "decision": "SIN ALERTA",
            "motivo": f"Zona sola {float(minutos_en_estado):g} min; aún faltan {faltan:g} min para el máximo de {limite}."}


def formatear(resultado: dict) -> str:
    """Resumen compacto en una línea para devolverle al LLM (ahorra tokens)."""
    return " | ".join(f"{k}={v}" for k, v in resultado.items())
