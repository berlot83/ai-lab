# Checklist de estudio de IA y NCA GENL

**Actualizado:** 28 de octubre de 2026.  
**Objetivo:** identificar qué trabajamos, qué conviene repasar y qué falta, separando el temario NVIDIA de nuestras prácticas complementarias.

El ejercicio de selección del próximo token está terminado para su alcance didáctico. El siguiente paso práctico es comparar configuraciones de generación con Mistral. La arquitectura Transformer fue explicada conceptualmente; no construimos ni entrenamos un Transformer.

## Cómo usar este checklist

### Origen de cada tema

- **NVIDIA · Objetivo:** corresponde a un objetivo explícito de la página oficial consultada.
- **NVIDIA · Lectura sugerida:** aparece como recurso recomendado. Eso no convierte cada producto o detalle del recurso en un objetivo independiente obligatorio.
- **Complementario:** ejercicio, herramienta o profundidad que elegimos para comprender y aplicar. Puede ayudar a un objetivo oficial aunque NVIDIA no nombre esa implementación concreta. No significa «seguro que no entra en el examen».

### Estados

- **Pendiente:** no hay trabajo registrado suficiente sobre ese punto.
- **Visto:** explicado conceptualmente.
- **Practicado parcial:** hay una práctica relacionada, pero no cubre todo el objetivo.
- **Practicado:** ejecutado y revisado para el alcance indicado.
- **Repaso pendiente:** volver a explicarlo o resolver otro caso para afianzarlo.

Las casillas `[x]` de la sección de prácticas significan que se realizó esa actividad concreta, no que el tema esté dominado. Las casillas `[ ]` indican acciones que faltan. Los estados de los objetivos oficiales se presentan por separado porque son más amplios que un ejercicio.

## 1 Mapa de los objetivos NVIDIA

Fuente: [NVIDIA NCA GENL, objetivos y preparación](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/), consultada el 28/10/2026.

Los nombres siguientes son **etiquetas abreviadas**, no una transcripción. Los identificadores son nuestros; su número sigue el orden de los objetivos dentro de cada dominio oficial. Consultar la fuente para el alcance completo. Los pesos corresponden al examen, no a nuestro porcentaje de avance. Algunos objetivos se repiten entre dominios y se conservan para no ocultar esa estructura.

### Fundamentos de ML e IA — 30%

| ID | NVIDIA · Objetivo abreviado | Estado registrado | Evidencia y siguiente paso |
|---|---|---|---|
| ML01 | Despliegue, escalabilidad, rendimiento y confiabilidad | Practicado parcial | Ollama y Qdrant usados localmente. Falta evaluar sistemáticamente el comportamiento de la aplicación bajo carga y fallos. |
| ML02 | Minería y visualización | Pendiente | Los logs del laboratorio no sustituyen un trabajo de análisis exploratorio. |
| ML03 | RAG, chatbots y resúmenes | Practicado parcial | Construimos un asistente RAG. No hay un ejercicio equivalente registrado para cada uno de los otros casos. |
| ML04 | Curación y embeddings RAG | Practicado parcial | Trabajamos fragmentos de documentación y sus embeddings. Falta ampliar el trabajo con fuentes reales y controles de calidad. |
| ML05 | Fundamentos, características, comparación y validación cruzada | Visto; repaso pendiente | Regresión, clasificación y particiones explicadas y practicadas en ejemplos pequeños. Falta ingeniería de características y validación cruzada. |
| ML06 | Bibliotecas Python y bases vectoriales | Practicado parcial | Qdrant utilizado; el laboratorio no demuestra todavía amplitud en las bibliotecas de NLP y datos. |
| ML07 | Lectura de investigación | Iniciado | Se entregó el PDF original de Attention Is All You Need. No damos por hecha su lectura completa ni su comprensión. |
| ML08 | Modelos de embeddings | Practicado parcial | Se usó BGE-M3 y se trabajó similitud coseno y recuperación. Repasar compatibilidad entre embeddings de consulta y documentos y comparar modelos. |
| ML09 | Prompting | Practicado parcial | Instrucciones, contexto, preguntas negativas y respuesta ante información ausente. Falta comparación sistemática de variantes. |
| ML10 | Análisis ML con bibliotecas | Pendiente | Los algoritmos manuales del laboratorio no equivalen todavía a entrenar/evaluar un análisis con bibliotecas de ML. |

### Análisis de datos — 14%

| ID | NVIDIA · Objetivo abreviado | Estado registrado | Evidencia y siguiente paso |
|---|---|---|---|
| DA01 | Minería y visualización | Pendiente | Mismo objetivo transversal que ML02; no duplicar artificialmente el avance. |
| DA02 | Comparación estadística de modelos | Visto; repaso pendiente | MAE/MSE y pérdida frente a métrica explicados y calculados en ejemplos. Falta comparar modelos sobre un mismo conjunto de evaluación. |
| DA03 | Análisis de datos | Pendiente | Preparar un pequeño dataset y un análisis reproducible. |
| DA04 | Gráficos de resultados | Pendiente | Falta generar e interpretar gráficos de un experimento propio. |
| DA05 | Relaciones, tendencias y factores | Pendiente | Falta un caso de análisis que identifique factores que alteran las conclusiones. |

### Experimentación — 22%

| ID | NVIDIA · Objetivo abreviado | Estado registrado | Evidencia y siguiente paso |
|---|---|---|---|
| EX01 | Entrenamiento y optimización | Practicado parcial; repaso pendiente | `training_validation.py` ajusta peso y sesgo con gradientes manuales. Falta optimización con bibliotecas, hiperparámetros y modelos de mayor escala. |
| EX02 | Datasets para pretraining, fine-tuning y RLHF | Visto parcial | Tokenización explicada; falta preparar un dataset con ese propósito. |
| EX03 | Pruebas hardware/software LLM | Practicado parcial | Pruebas funcionales manuales del RAG y de selección de tokens. Faltan pruebas repetibles y mediciones de recursos. |
| EX04 | Evaluación tecnológica | Visto parcial | Se discutieron alternativas de modelos y ejecución local. Falta una comparación documentada con criterios y mediciones. |
| EX05 | Datos humanos y RLHF | Pendiente | No hay ejercicio registrado de feedback humano o alineación. |
| EX06 | Benchmarks y refinamiento | Pendiente | Falta un conjunto de pruebas con criterios de aceptación y comparación de versiones. |
| EX07 | Experimentos de modelos/pipelines | Practicado parcial | Cambiamos documentos del RAG y comprobamos respuestas; probamos temperatura. Falta consolidar un protocolo reproducible. |

### Desarrollo de software — 24%

| ID | NVIDIA · Objetivo abreviado | Estado registrado | Evidencia y siguiente paso |
|---|---|---|---|
| SW01 | Despliegue y evaluación operativa | Practicado parcial | Relacionado con ML01; ejecución local no equivale a validación operativa completa. |
| SW02 | Aplicaciones LLM | Practicado parcial | Relacionado con ML03; RAG local implementado. |
| SW03 | Bibliotecas NLP y vectores | Practicado parcial | Relacionado con ML06; Qdrant y llamadas a Ollama. |
| SW04 | Selección de componentes | Practicado parcial | Elegimos modelo de embeddings, generador y almacenamiento vectorial para el laboratorio. Falta justificar decisiones con requisitos y mediciones. |
| SW05 | Monitoreo de procesos | Practicado parcial | Logs y comprobación de inserciones/cambios. Falta observabilidad y seguimiento sistemático de experimentos. |
| SW06 | Análisis ML programático | Pendiente | Relacionado con ML10. |
| SW07 | Componentes y scripts | Practicado | Scripts propios y funciones revisadas mediante código/salidas compartidas. Quedan robustez y pruebas automatizadas por ampliar. |

### IA confiable — 10%

| ID | NVIDIA · Objetivo abreviado | Estado registrado | Evidencia y siguiente paso |
|---|---|---|---|
| TA01 | Principios éticos | Pendiente | Falta un bloque dedicado con casos. |
| TA02 | Privacidad y consentimiento | Pendiente | No damos por cubierto este objetivo por trabajar con documentación ficticia. |
| TA03 | Tecnologías de confianza | Pendiente | Falta evaluar mecanismos concretos de protección y control. |
| TA04 | Reducción de sesgos | Pendiente | Falta identificar y evaluar sesgos en datos y respuestas. |

## 2 Temas y prácticas de nuestro laboratorio

### Bases de ML

Estos contenidos desarrollan fundamentos del plan NVIDIA, pero la selección de ejemplos y su profundidad son nuestras.

- [x] **Visto:** regresión y clasificación.
- [x] **Visto:** train/validation/test y sobreajuste/subajuste.
- [x] **Visto:** diferencia entre función de pérdida y métrica.
- [x] **Visto:** MAE, MSE, accuracy, precision y recall.
- [ ] **Repaso pendiente:** resolver ejemplos nuevos sin mirar la explicación.
- [ ] **Pendiente:** clasificación binaria/multiclase, matriz de confusión y ROC/AUC con un caso concreto.
- [ ] **Pendiente:** validación cruzada, fuga de datos y comparación de modelos.

No se asigna una fecha exacta de repaso completado a los primeros conceptos: el historial confirma que se vieron y que Axel pidió retomarlos, no una evaluación reciente de todos ellos.

### RAG y recuperación

- [x] **Practicado · Complementario aplicado a NVIDIA:** búsqueda inicial por texto.
- [x] **Practicado · Complementario aplicado a NVIDIA:** embeddings con Ollama y BGE-M3.
- [x] **Practicado:** similitud coseno y selección top-k de fragmentos.
- [x] **Practicado:** Qdrant, colección `documentation`, vectores y payloads.
- [x] **Practicado:** omitir fragmentos sin cambios al actualizar; pruebas registradas el 23/10.
- [x] **Practicado:** recuperar texto y enviarlo como contexto a Mistral.
- [x] **Practicado parcial:** preguntas negativas, información ausente y documentación actualizada.
- [x] **Visto:** límites de contexto y lost in the middle.
- [x] **Visto:** RAG frente a fine-tuning.
- [ ] **Pendiente:** evaluar recuperación y generación por separado con respuestas esperadas.
- [ ] **Pendiente:** pruebas automatizadas, regresiones y métricas por versión del pipeline.

Qdrant, BGE-M3, Mistral y Ollama son elecciones del laboratorio. No se presentan como productos exigidos específicamente por NVIDIA.

### Transformers

**NVIDIA · Lectura sugerida:** [Attention Is All You Need, primera versión de 2017](https://arxiv.org/pdf/1706.03762v1). La explicación y los ejercicios concretos son complementarios.

- [x] **Visto:** tokens, embeddings y posiciones.
- [x] **Visto:** atención causal y múltiples cabezas.
- [x] **Visto:** MLP, residuales, normalización y repetición de bloques.
- [x] **Visto:** logits de salida y generación autorregresiva.
- [x] **Practicado acotado:** cuentas ilustrativas de residual y normalización.
- [ ] **Repaso pendiente:** reconstruir el recorrido de un texto hasta los logits con palabras propias.
- [ ] **Pendiente opcional de profundización:** cálculo concreto de atención con Q, K y V.
- [ ] **Pendiente:** lectura guiada de las partes relevantes del paper.

No programamos el cálculo de logits a partir del contexto ni entrenamos un Transformer. Construir uno completo desde cero no se establece como requisito de este plan.

### Selección del próximo token

**Complementario aplicado a fundamentos/programación:** implementación manual para entender la generación. La página oficial consultada no exige explícitamente reproducir este archivo ni calcular exponentes a mano.

- [x] **Practicado:** logits escritos a mano y asociación token–puntaje.
- [x] **Practicado:** softmax y suma de probabilidades.
- [x] **Practicado:** greedy y conservación del índice original.
- [x] **Practicado:** intervalos acumulados y sampling ponderado.
- [x] **Practicado:** 100 sorteos, distinguiendo frecuencias observadas de probabilidades — 27/10.
- [x] **Practicado:** temperatura 0.5, 1 y 2 — 27/10.
- [x] **Practicado acotado:** ordenamiento y corte top-k mediante `candidates[:2]` — 27/10.
- [x] **Practicado:** top-p que incluye al candidato que alcanza o cruza el umbral — 27/10.
- [x] **Repasado:** propósito del programa; separación entre logits inventados y logits de un modelo real — 27–28/10.
- [x] **Repasado:** greedy/sampling y top-k/top-p con ejemplos nuevos — 28/10.
- [x] **Repasado:** renormalizar 0.40 y 0.10 a 0.80 y 0.20 — 28/10.
- [x] **Practicado:** renormalización manteniendo las posiciones originales — 28/10.
- [x] **Practicado:** sampling con probabilidades filtradas — 28/10; salida final: 7 grises y 3 azules.
- [ ] **Repaso futuro breve:** explicar el propósito de cada etapa sin depender del código.

**Cierre de alcance:** `token_selection.py` completó el ejercicio didáctico. No continuar agregando funcionalidades por inercia. Las pruebas mostradas no constituyen validación de producción para cualquier entrada, extremos numéricos o parámetros inválidos.

### Herramientas y ampliaciones pendientes

- [ ] **Complementario aplicado a NVIDIA:** comparar configuraciones reales de Mistral manteniendo fijo el prompt.
- [ ] **Complementario aplicado a NVIDIA:** NumPy y un framework de redes neuronales en ejercicios pequeños.
- [ ] **Complementario aplicado a NVIDIA:** gradientes, backpropagation, optimizadores, learning rate, épocas y batches.
- [ ] **NVIDIA · Lectura sugerida:** LoRA, TensorRT, NeMo y Triton; estudiar su función y relación con los objetivos, no memorizar catálogos.
- [ ] **Complementario aplicado a NVIDIA:** cuantización, memoria, latencia y throughput mediante mediciones propias.
- [ ] **Complementario aplicado a NVIDIA:** guardrails, fallos del RAG e instrucciones maliciosas en documentos recuperados.
- [ ] **Preparación de examen:** preguntas por dominio, registro de errores y simulacros.

La referencia a lecturas sugeridas se apoya en la [lista oficial de recursos](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/). Su estado aquí es pendiente salvo evidencia explícita anterior.

## 3 Próximos pasos acordados

1. Comparar generación con Mistral usando un mismo prompt y configuraciones controladas.
2. Evaluar el RAG con preguntas y respuestas esperadas, separando errores de recuperación y generación.
3. Ampliar cobertura hacia datos, entrenamiento, evaluación, despliegue e IA confiable.
4. Intercalar repasos breves de ML y Transformers sin reiniciar todo el recorrido.
5. Revisar nuevamente los objetivos oficiales antes de programar el examen.

## 4 Registro de repaso reutilizable

Hacer una pregunta por vez y pedir la explicación antes de mostrar la respuesta. Una duda de sintaxis o una consigna no leída no se registra automáticamente como dificultad conceptual.

| Fecha | Tema | Qué pude explicar o resolver | Qué falta reforzar | Próxima comprobación |
|---|---|---|---|---|
| 28/10/2026 | Greedy y sampling | Greedy elige el máximo; sampling permite candidatos menos probables | Mantener separadas probabilidad por sorteo y frecuencia observada | Otro ejemplo con distribución fija |
| 28/10/2026 | Top-k y top-p | k=2 excluye el tercero; p=0.80 conserva 60%+30% | Recordar ordenar primero por probabilidad | Distribución donde k y p elijan cantidades diferentes |
| 28/10/2026 | Renormalización | 0.40/0.50=0.80; 0.10/0.50=0.20 | Ubicar en código dónde se calculó el denominador | Explicar `probability / cumulative` |
| 28/10/2026 | Pipeline didáctico | Ejecutó filtrado, renormalización y sampling integrado | Revisión espaciada de su propósito | Relacionarlo con una configuración de Mistral |
| | | | | |

## 5 Regla para actualizar este archivo

Agregar evidencia breve cuando se complete una actividad: código ejecutado, salida revisada o ejemplo resuelto. Cambiar «Visto» por «Practicado» solo cuando corresponda. Mantener los repasos como acciones separadas. No calcular avance dividiendo casillas marcadas entre casillas totales: las actividades y los objetivos tienen alcances diferentes.
