"""Agente de la Unidad 5: create_agent + 2 herramientas + memoria por thread_id."""

import logging
import os
import time
import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from .config import MODELO_GEMINI
from .tools import HERRAMIENTAS

PROMPT_SISTEMA = """Eres el asistente de supervisión de SafeRest 360 para el administrador de un restaurante.
- Si el usuario solo saluda o hace una pregunta general, responde sin usar herramientas.
- Usa clasificar_zona solo si el mensaje trae un identificador de imagen (img-xxxxxx).
- Usa evaluar_regla_zona para decidir si se debe avisar. Si el estado viene de clasificar_zona, pásale su confianza.
- Convierte la hora a formato de 24 horas (HH:MM). Si la zona o la hora ya se dijeron antes en esta conversación, reutilízalas.
- No inventes zonas, horas ni resultados: si un dato no aparece en la conversación, pregúntalo.
- Un aviso es para revisar, nunca una acusación; la decisión final la toma una persona.
Responde en español y en máximo 4 frases."""


def construir_agente(checkpointer=None):
    """Crea el agente con memoria. Requiere GOOGLE_API_KEY en el archivo .env."""
    load_dotenv()
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta GOOGLE_API_KEY. Copien .env.example a .env y pongan su clave de Google AI Studio.")
    llm = ChatGoogleGenerativeAI(model=MODELO_GEMINI, api_key=api_key, temperature=0)
    return create_agent(
        model=llm,
        tools=HERRAMIENTAS,
        system_prompt=PROMPT_SISTEMA,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


def nuevo_thread_id() -> str:
    return f"sesion-{uuid.uuid4().hex[:8]}"


def mensajes_del_ultimo_turno(mensajes):
    """Con memoria, invoke() devuelve TODO el historial del hilo; aquí se toma solo el último turno."""
    indices_humanos = [i for i, m in enumerate(mensajes) if getattr(m, "type", None) == "human"]
    return mensajes[indices_humanos[-1] + 1:] if indices_humanos else mensajes


def herramientas_invocadas(mensajes_turno):
    """Trayectoria del turno leída de los tool_calls de los mensajes del agente (no del texto final)."""
    return [
        {"nombre": llamada["name"], "args": llamada["args"]}
        for mensaje in mensajes_turno
        if getattr(mensaje, "tool_calls", None)
        for llamada in mensaje.tool_calls
    ]


def tokens_del_turno(mensajes_turno) -> dict:
    """Suma los tokens que reporta la API en cada llamada al LLM del turno."""
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for mensaje in mensajes_turno:
        uso = getattr(mensaje, "usage_metadata", None) or {}
        for clave in total:
            total[clave] += int(uso.get(clave, 0) or 0)
    return total


def preguntar(agente, mensaje: str, thread_id: str, reintentos: int = 3, espera_s: float = 20.0) -> dict:
    """Envía un mensaje al hilo thread_id. Reintenta si la API responde por límite de cuota (429)."""
    config = {"configurable": {"thread_id": thread_id}}
    for intento in range(reintentos + 1):
        try:
            resultado = agente.invoke({"messages": [{"role": "user", "content": mensaje}]}, config=config)
            break
        except Exception as error:  # la API gratuita devuelve 429 / RESOURCE_EXHAUSTED al superar la cuota
            texto = str(error)
            if intento < reintentos and ("429" in texto or "RESOURCE_EXHAUSTED" in texto):
                time.sleep(espera_s * (intento + 1))
                continue
            raise
    turno = mensajes_del_ultimo_turno(resultado["messages"])
    return {
        "respuesta": resultado["messages"][-1].text,
        "herramientas": herramientas_invocadas(turno),
        "salidas_herramientas": [m.content for m in turno if getattr(m, "type", None) == "tool"],
        "tokens": tokens_del_turno(turno),
        "mensajes_en_hilo": len(resultado["messages"]),
    }
