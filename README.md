<!--
PLANTILLA DEL README — LÉANLA ANTES DE ESCRIBIR
- Esta plantilla trae SOLO la estructura obligatoria (7 secciones) y comentarios GUÍA como este.
- El texto de las secciones 1 a 6 lo escriben ustedes, con sus palabras. Si es transcripción o paráfrasis
  de una IA, o cita cifras que no están en los notebooks, ese criterio queda con máximo 2.0.
- Toda cifra que citen debe aparecer en las SALIDAS GUARDADAS de los notebooks o en una gráfica de assets/.
- Borren todos los comentarios GUÍA y los marcadores [COMPLETAR] antes de entregar.
-->

# SafeRest 360 · Vigía de zonas

<!-- GUÍA: título definitivo del proyecto. -->

## Sección 1 — Identificación, ruta y propuesta de valor

**Integrantes:**
- Wilson Sebastián Moreno Sánchez
- Michael Stevent Chavarro Nope
- Paula Selene Becerra Morales

**Ruta:** Ruta 1 — Visión por Computador + Agente de Diagnóstico / Inspección

**Dataset:** subconjunto estilo Visual Wake Words construido desde COCO val2017 (ver `notebooks/entrenamiento_u4.ipynb`, sección 2).

<!-- GUÍA: expliquen con sus palabras por qué este problema necesita una red neuronal (U4) + un agente (U5)
y no solo un prompt a un LLM. Pistas para pensarlo (no para copiarlas): ¿qué pasaría con la privacidad y
los tokens si le mandaran la imagen al LLM? ¿Un LLM aplica siempre igual una regla de horario?
¿Qué aporta tener la probabilidad de la red? -->

[COMPLETAR]

| Integrante | Componente que desarrolló |
|---|---|
| Wilson Sebastián Moreno Sánchez | [COMPLETAR] |
| Michael Stevent Chavarro Nope | [COMPLETAR] |
| Paula Selene Becerra Morales | [COMPLETAR] |

## Sección 2 — Arquitectura del sistema (diagrama propio)

<!-- GUÍA: el diagrama lo hacen ustedes (draw.io, Excalidraw, Canva…) y lo guardan en assets/.
Debe mostrar: interfaz Streamlit → agente (create_agent) → @tool #1 clasificar_zona (red neuronal local)
y @tool #2 evaluar_regla_zona (reglas) → memoria InMemorySaver por thread_id.
Archivos reales del código: app.py, src/agent.py, src/tools.py, src/vision.py, src/reglas.py. -->

![Arquitectura](assets/arquitectura.png)

[COMPLETAR]

## Sección 3 — Ingeniería del modelo de red neuronal (Unidad 4) y análisis de un fallo real

<!-- GUÍA:
- Arquitectura e hiperparámetros: celdas de las secciones 4 y 5 del notebook (summary(), EPOCAS, TAMANO_LOTE,
  Dropout, tamaño de imagen 160, preprocesamiento OpenCV con letterbox en src/vision.py).
- Comparación base vs. mejorado: tabla impresa en la sección 7 y resumen de la sección 10 del notebook.
- Fallo real: elijan UNO de los casos impresos en la sección 9 (índice, archivo, real, predicción, p_persona)
  y analícenlo mirando la imagen en assets/fallos/. Usen también la tabla de desempeño por tamaño de persona
  y por iluminación. -->

![Curvas CNN base](assets/curvas_cnn_base.png)
![Curvas MobileNetV2](assets/curvas_mobilenetv2.png)
![Comparación en validación](assets/comparacion_validacion.png)
![Matrices de confusión](assets/matrices_confusion.png)
![Fallos analizados](assets/analisis_fallos.png)
![Desempeño por condición](assets/desempeno_por_condicion.png)

[COMPLETAR]

## Sección 4 — Diseño del agente y matriz de evaluación (Unidad 5)

<!-- GUÍA:
- Las 2 herramientas y cómo redactaron sus docstrings: src/tools.py.
- Memoria: InMemorySaver + thread_id (src/agent.py) y botón de reinicio en la app.
- Cuidado de tokens/cuota: lo que hicieron y las cifras de tokens impresas en evaluacion_agente_u5.ipynb.
- Matriz de evaluación: copien la tabla impresa al final del notebook (también está en assets/evaluacion_agente.csv). -->

[COMPLETAR]

| Caso | Qué prueba | Esperadas | Invocadas | Trayectoria | Decisión esperada | Decisión obtenida | Memoria | Resultado |
|---|---|---|---|---|---|---|---|---|
| 1 | [COMPLETAR] | | | | | | | |
| 2 | | | | | | | | |
| 3 | | | | | | | | |
| 4 | | | | | | | | |

## Sección 5 — Bitácora de Vibe Coding y razonamiento crítico

<!-- GUÍA: cada episodio necesita su captura en assets/bitacora/ y, si la herramienta lo permite, el enlace
compartido de la conversación. Declarar uso de IA sin captura descuenta 0.5.
Documenten dónde la IA propuso código con errores, APIs desactualizadas, fugas de datos o malas prácticas,
y cómo lo detectaron y corrigieron ustedes. -->

| # | Herramienta de IA | Qué le pidieron | Problema que encontraron en lo que propuso | Cómo lo corrigieron | Evidencia |
|---|---|---|---|---|---|
| 1 | [COMPLETAR] | | | | ![captura](assets/bitacora/episodio_1.png) · [enlace]() |

## Sección 6 — Demostración de la interfaz

<!-- GUÍA: capturas en assets/ y enlace al video (máximo 3 minutos, YouTube no listado o Google Drive).
El video debe mostrar los casos de evaluación, el caso de fallo de la Sección 3 (está en
"Ejemplos del notebook" dentro de la app), la salida de la red y las herramientas invocadas. -->

**Video:** [COMPLETAR enlace]

![App](assets/app_1.png)

## Sección 7 — Guía de instalación y ejecución reproducible

<!-- GUÍA: verifiquen estos pasos en un computador limpio y ajústenlos si algo cambia. -->

```bash
# 1. Clonar el repositorio
git clone https://github.com/paubm21/saferest360-vigia-zonas.git
cd saferest360-vigia-zonas

# 2. Crear y activar un entorno virtual (Python 3.12)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar la API key de Google AI Studio
cp .env.example .env             # Windows: copy .env.example .env
#    editar .env y reemplazar tu_clave_aqui por la clave

# 5. (Opcional) Reentrenar: ejecutar notebooks/entrenamiento_u4.ipynb
#    descarga COCO val2017 (~1 GB) en data/ y regenera models/ y assets/

# 6. Ejecutar la aplicación
streamlit run app.py
```

Los pesos entrenados ya están en `models/` (`modelo_info.json` indica cuál usa la app).
