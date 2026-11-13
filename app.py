"""SafeRest 360 · Vigía de zonas — app de Streamlit (Taller Unidades 4 y 5).

Ejecutar:  streamlit run app.py
"""

import hashlib
import json

import cv2
import streamlit as st

from src import tools
from src.agent import construir_agente, nuevo_thread_id, preguntar
from src.config import DIR_EJEMPLOS, DIR_FALLOS, RUTA_INFO_MODELO
from src.vision import ClasificadorZona, leer_imagen_bgr

st.set_page_config(page_title="SafeRest 360 · Vigía de zonas", layout="wide")


# ---------------------------------------------------------------------------
# Recursos en caché: se cargan una sola vez, no en cada mensaje
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando la red neuronal…")
def cargar_clasificador():
    return ClasificadorZona()


@st.cache_resource(show_spinner="Construyendo el agente…")
def cargar_agente():
    return construir_agente()


def info_modelo() -> dict:
    if RUTA_INFO_MODELO.exists():
        return json.loads(RUTA_INFO_MODELO.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Estado de la sesión
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = nuevo_thread_id()
if "chat" not in st.session_state:
    st.session_state.chat = []          # [{"rol", "texto", "herramientas", "salidas", "tokens"}]
if "imagen" not in st.session_state:
    st.session_state.imagen = None      # {"hash", "id", "nombre", "rgb", "pred"}

try:
    clasificador = cargar_clasificador()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()
tools.configurar_clasificador(clasificador)

try:
    agente = cargar_agente()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

# ---------------------------------------------------------------------------
# Barra lateral: memoria (thread_id) y datos del modelo
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Sesión del agente")
    st.code(st.session_state.thread_id, language=None)
    if st.button("Reiniciar conversación (nuevo thread_id)"):
        st.session_state.thread_id = nuevo_thread_id()
        st.session_state.chat = []
        st.rerun()
    st.caption("La memoria del agente vive en un InMemorySaver indexado por este thread_id.")

    st.header("Red neuronal (Unidad 4)")
    datos = info_modelo()
    if datos:
        st.write(f"Configuración: **{datos.get('configuracion', '—')}**")
        metricas = datos.get("metricas_prueba", {})
        if metricas:
            st.write(f"Exactitud en prueba: **{metricas.get('accuracy', 0):.3f}**")
            st.write(f"Recall clase ocupada: **{metricas.get('recall', 0):.3f}**")
        st.caption("Cifras leídas de models/modelo_info.json, generado por el notebook de entrenamiento.")
    st.caption("Las imágenes se procesan localmente; al LLM solo llega un identificador y una línea de texto.")

st.title("SafeRest 360 · Vigía de zonas")
st.caption("Red neuronal que detecta si una zona está ocupada o vacía + agente que aplica las reglas del restaurante.")

col_imagen, col_chat = st.columns([2, 3], gap="large")

# ---------------------------------------------------------------------------
# Columna 1: imagen de la zona y salida directa de la red neuronal
# ---------------------------------------------------------------------------
with col_imagen:
    st.subheader("1. Imagen de la zona")
    ejemplos = sorted(list(DIR_FALLOS.glob("*.jpg")) + list(DIR_EJEMPLOS.glob("*.jpg")))
    origen = st.radio("Origen", ["Subir imagen", "Ejemplos del notebook"], horizontal=True)

    contenido, nombre = None, None
    if origen == "Subir imagen":
        archivo = st.file_uploader("Imagen JPG o PNG", type=["jpg", "jpeg", "png"])
        if archivo is not None:
            contenido, nombre = archivo.getvalue(), archivo.name
    elif ejemplos:
        elegido = st.selectbox("Ejemplo", ejemplos, format_func=lambda p: f"{p.parent.name}/{p.name}")
        contenido, nombre = elegido.read_bytes(), f"{elegido.parent.name}/{elegido.name}"
    else:
        st.info("Aún no hay ejemplos: el notebook de entrenamiento los guarda en assets/fallos y assets/ejemplos.")

    if contenido is not None:
        huella = hashlib.md5(contenido).hexdigest()
        actual = st.session_state.imagen
        if actual is None or actual["hash"] != huella:
            imagen_bgr = leer_imagen_bgr(contenido)
            st.session_state.imagen = {
                "hash": huella,
                "id": tools.registrar_imagen(imagen_bgr, etiqueta=nombre),
                "nombre": nombre,
                "rgb": cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB),
                "pred": clasificador.predecir(imagen_bgr),
            }

    actual = st.session_state.imagen
    if actual is not None:
        st.image(actual["rgb"], caption=actual["nombre"])
        st.markdown("**Salida directa de la red neuronal**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Estado", actual["pred"]["estado"])
        m2.metric("P(persona)", f"{actual['pred']['p_persona']:.2f}")
        m3.metric("Confianza", f"{actual['pred']['confianza']:.2f}")
        st.caption(f"Modelo: {actual['pred']['modelo']} · ID para el agente: `{actual['id']}`")

# ---------------------------------------------------------------------------
# Columna 2: conversación con el agente
# ---------------------------------------------------------------------------
with col_chat:
    st.subheader("2. Conversación con el agente")
    adjuntar = st.checkbox(
        "Adjuntar el ID de la imagen actual a mi próximo mensaje",
        value=st.session_state.imagen is not None,
        disabled=st.session_state.imagen is None,
    )

    for turno in st.session_state.chat:
        with st.chat_message("user" if turno["rol"] == "usuario" else "assistant"):
            st.markdown(turno["texto"])
            if turno["rol"] == "agente":
                if turno["herramientas"]:
                    nombres = " → ".join(f"`{h['nombre']}`" for h in turno["herramientas"])
                    st.caption(f"Herramientas invocadas: {nombres}")
                    with st.expander("Detalle de las herramientas"):
                        for h, salida in zip(turno["herramientas"], turno["salidas"]):
                            st.markdown(f"**{h['nombre']}**  \nargs: `{h['args']}`  \nsalida: `{salida}`")
                else:
                    st.caption("Herramientas invocadas: ninguna")
                st.caption(f"Tokens del turno: {turno['tokens'].get('total_tokens', 0)}")

    pregunta = st.chat_input("Ej.: Revisa la bodega, son las 23:30. ¿Hay que avisar?")
    if pregunta:
        texto = pregunta
        if adjuntar and st.session_state.imagen is not None:
            texto = f"{pregunta}\n(imagen adjunta: {st.session_state.imagen['id']})"
        st.session_state.chat.append({"rol": "usuario", "texto": texto})
        with st.spinner("El agente está pensando…"):
            try:
                r = preguntar(agente, texto, st.session_state.thread_id)
                st.session_state.chat.append({
                    "rol": "agente", "texto": r["respuesta"], "herramientas": r["herramientas"],
                    "salidas": r["salidas_herramientas"], "tokens": r["tokens"],
                })
            except Exception as error:
                st.session_state.chat.append({
                    "rol": "agente", "texto": f"Error al consultar el agente: {error}",
                    "herramientas": [], "salidas": [], "tokens": {},
                })
        st.rerun()
