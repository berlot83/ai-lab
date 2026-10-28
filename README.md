# ai-lab: guía de repaso del laboratorio

**Estado documental: 28 de octubre de 2026.**

Este README documenta el inventario real del repositorio revisado el 28/10/2026. Reúne ejercicios deliberadamente explícitos para estudiar Python, fundamentos de ML, embeddings, recuperación aumentada por generación (RAG) y selección de tokens. Su objetivo es permitir el repaso; no es una aplicación de producción.

## 1. Qué hay en cada archivo

| Archivo | Para qué sirve | Alcance verificado de esta documentación |
|---|---|---|
| `documentation_assistant.py` | Recuperación léxica simple y respuesta con Mistral. | Código actual revisado. |
| `embedding_example.py` | Recorrido RAG didáctico: embeddings, similitud coseno, Qdrant y generación. | Código actual revisado. |
| `embeddings.json` | Tres fragmentos y sus vectores BGE-M3 precalculados. | Estructura, modelo, cantidad y dimensión verificados. |
| `index_documents.py` | Indexación incremental de cuatro documentos en Qdrant. | Código actual revisado; no se ejecutó para no mutar Qdrant. |
| `ask_documentation.py` | Recupera los dos fragmentos más cercanos y genera una respuesta restringida a ellos. | Código actual revisado; no se hicieron llamadas externas. |
| `token_selection.py` | Softmax, temperatura, greedy, sampling, filtro top-p y renormalización. | Código actual y salida de estudio documentados abajo. |
| `training_validation.py` | Regresión lineal manual con train/validation/test y descenso de gradiente. | Código actual revisado. |
| `delivery_metrics.py` | Cálculo manual de MAE y MSE para regresión. | Código actual revisado. |
| `delivery_accuracy.py` | Cálculo manual de accuracy para clasificación. | Código actual revisado. |

`.DS_Store` es metadato de macOS y no forma parte del laboratorio. Los scripts solo importan módulos de la biblioteca estándar, pero varios requieren servicios locales.

---

## Requisitos, ejecución y orden de repaso

Para los ejemplos locales alcanza con Python 3.10 o posterior. Los ejercicios RAG requieren Ollama en `localhost:11434` con `bge-m3` y `mistral:latest`; los que usan almacenamiento vectorial requieren además Qdrant en `localhost:6333`. Ejecutar desde la raíz porque `embedding_example.py` abre `embeddings.json` con una ruta relativa.

```bash
# Sin servicios externos ni cambios persistentes
python3 delivery_accuracy.py
python3 delivery_metrics.py
python3 training_validation.py
python3 token_selection.py

# Requieren servicios locales
python3 documentation_assistant.py
python3 index_documents.py
python3 ask_documentation.py
```

Orden recomendado: métricas y entrenamiento manual; selección de tokens; recuperación léxica; recorrido de embeddings; indexación incremental; consulta RAG. `index_documents.py` puede crear o actualizar la colección `documentation`: ejecutarlo solo cuando se quiera producir ese cambio. No se ejecutaron llamadas externas para preparar este README.

## `documentation_assistant.py`

Busca por subcadena dentro de tres fragmentos ficticios. Si encuentra contexto, pide una pregunta, construye un prompt y llama a `mistral:latest`; si no, informa que no hubo coincidencias.

### `retrieve_chunks(documents: list[str], query: str)`

- **Parámetros:** `documents`, fragmentos; `query`, texto buscado.
- **Retorno:** lista de fragmentos que contienen la consulta sin distinguir mayúsculas. Una consulta vacía después de `strip()` devuelve `[]`.
- **Variables:** `normalized_query` quita espacios laterales; `chunks` acumula resultados; `doc` es el fragmento actual.

Variables del flujo: `document_chunks` contiene la documentación; `search_term` y `question` vienen de consola; `retrieved_chunks` y `context` forman la evidencia; `prompt` agrega instrucciones; `payload`, `payload_text` y `request_body` serializan la petición; `request`, `response_bytes`, `response_text` y `result` representan la llamada y respuesta HTTP. La recuperación es textual, no semántica.

## `embedding_example.py`

Es un recorrido RAG monolítico: pregunta → embedding → consulta top-2 en Qdrant → contexto → Mistral. La llamada `# index_documents()` está comentada; el flujo normal no indexa, aunque sí consulta Ollama y Qdrant.

### Funciones

- `generate_embedding(text) -> list[float]`: envía `text` a `/api/embed` con `bge-m3` y devuelve el primer vector. `payload`, `json_payload`, `payload_bytes`, `request` y `response_*` son las etapas de serialización y HTTP.
- `dot_product(a, b) -> float`: valida igual dimensión y suma `a[i] * b[i]`. `result` es el acumulador.
- `vector_norm(vector) -> float`: devuelve la norma euclídea usando el producto punto.
- `cosine_similarity(a, b) -> float`: divide el producto por ambas normas; rechaza vectores de norma cero. `product`, `norm_a` y `norm_b` conservan las partes del cálculo.
- `embeddings(chunks) -> list[tuple[str, list[float]]]`: devuelve pares `(texto, vector)`; `embedded_chunks` es el acumulador.
- `index_documents() -> dict`: lee `embeddings.json`, valida el modelo, crea `documentation` si falta y carga los puntos. Devuelve la respuesta del upsert. `collection_exists`, `collection_payload`, `points`, `point_id`, `chunk` y `vector` controlan la carga. A diferencia de `index_documents.py`, no omite textos sin cambios.
- `retrieve_chunks(_question) -> str`: embebe la pregunta, consulta Qdrant con `limit=2` y devuelve textos unidos. `question_vector`, `query_payload`, `retrieved_points`, `context_chunks`, `chunk` y `score` describen la recuperación.
- `generate_answer(instructions, context, quest)`: arma el prompt y llama a Mistral. Imprime y retorna implícitamente `None`.

`document_chunks` son textos didácticos; `question` pregunta por un dato ausente para comprobar que el modelo no lo invente; `context` recibe la recuperación final.

## `embeddings.json`

Artefacto de datos con `model: "bge-m3"` y `chunks`: tres pares `[texto, vector]`. Cada vector tiene 1024 dimensiones y los textos coinciden con `embedding_example.py`. Editar un texto sin regenerar su vector lo deja incoherente; consultas y documentos deben usar embeddings compatibles.

## `index_documents.py`

Implementa la indexación incremental: valida IDs → asegura la colección → lee payloads existentes → omite textos iguales → genera embeddings nuevos o modificados → hace upsert.

### Constantes y datos

`QDRANT_URL` y `OLLAMA_URL` apuntan a servicios locales; `COLLECTION`, `MODEL` y `VECTOR_SIZE` fijan `documentation`, `bge-m3` y 1024; `COLLECTION_URL` deriva la URL; `DOCUMENTS` contiene pares `(ID estable, texto)`. El ID identifica al documento y no debe cambiar por editar su texto.

### Funciones

- `request_json(method, url, body=None) -> dict`: serializa `body`, hace HTTP y devuelve JSON; propaga errores. `data` son bytes o `None`.
- `ensure_collection() -> None`: crea la colección solo ante 404. Si existe con tamaño o distancia incompatibles, lanza `ValueError` y no la modifica. `result` y `vectors` conservan la configuración.
- `retrieve_documents(ids) -> dict[int, dict]`: devuelve `ID → payload` para puntos existentes; con lista vacía retorna `{}`. Pide payload pero no vectores.
- `generate_embedding(text) -> list[float]`: llama a BGE-M3 y valida exactamente 1024 dimensiones.
- `index_documents(documents) -> None`: valida IDs únicos, compara textos, acumula `points`, cuenta `skipped` y hace un upsert si hay cambios. Conserva otros campos del payload.

El bloque `if __name__ == "__main__"` evita indexar al importar. Quitar un documento de `DOCUMENTS` no lo borra. Cambiar el modelo exige regenerar todos los embeddings; el script no detecta ese cambio automáticamente.

## `ask_documentation.py`

`generate_embedding(text)` llama a BGE-M3 y devuelve el vector. `retrieve_chunks(_question)` consulta los dos puntos más cercanos (`limit=2`, distinto del top-k de tokens), imprime score y texto, y retorna contexto. `generate_answer(instructions, context, quest)` llama a Mistral, imprime y retorna `None` implícitamente.

`question` es fija; `context` recibe los dos fragmentos. Las instrucciones prohíben confundir timeout, intervalo y cantidad de intentos, y exigen reconocer datos ausentes. El flujo vive en el nivel principal: importar el archivo también dispara llamadas externas.

## `training_validation.py`

Regresión lineal manual `y = weight * x + bias`, con particiones de entrenamiento, validación y test.

- `predict(x, weight, bias)`: devuelve la predicción.
- `calculate_mse(data, weight, bias)`: devuelve el error cuadrático medio; `total_error` acumula cuadrados.
- `calculate_mae(data, weight, bias)`: devuelve el error absoluto medio.

`training_data`, `validation_data` y `test_data` tienen roles distintos. `weight` y `bias` son parámetros; `learning_rate` controla el paso; `epochs` la cantidad de pasadas; `weight_gradient` y `bias_gradient` son gradientes promedio; `error` es predicción menos esperado; `training_error`, `validation_error`, `test_error` y `validation_mae` son métricas. No hay batches, barajado, regularización ni autodiferenciación.

## `delivery_metrics.py`

Calcula MAE y MSE sin funciones. `actual_times` son valores reales; `predicted_times` queda como referencia pero no participa en el bucle actual; `predicted_times_large_error` sí se usa. `error` conserva el signo; `total_absolute_error` y `total_squared_error` acumulan; `mae` y `mse` promedian. MAE queda en minutos y MSE en minutos cuadrados.

## `delivery_accuracy.py`

Calcula accuracy sin funciones. `ON_TIME` y `LATE` son etiquetas; `actual_labels` es la verdad; `predicted_labels` la predicción; `correct_predictions` cuenta coincidencias; `actual` y `predicted` son el par actual; `accuracy` divide aciertos por cantidad de etiquetas reales. Con los datos actuales hay dos aciertos de tres. El script no valida longitudes iguales y accuracy puede engañar con clases desbalanceadas.

## Relación entre los archivos RAG

```text
embeddings.json ──> embedding_example.py
                           │
Ollama/BGE-M3 ──> index_documents.py ──> Qdrant/documentation
                                              │ top-2
                                              ▼
                    ask_documentation.py / embedding_example.py
                                              │ contexto
                                              ▼
                                       Ollama/Mistral
```

`documentation_assistant.py` es el antecedente sin embeddings ni Qdrant. `index_documents.py` es la variante incremental recomendada; `embedding_example.py` conserva el recorrido monolítico para estudiar sus piezas.

## 2. El problema que resuelve `token_selection.py`

Un modelo generativo debe decidir qué token agregar a continuación. Hay dos partes distintas:

1. **El modelo procesa el contexto y calcula logits**, es decir, puntajes para los tokens candidatos.
2. **El sistema de generación elige el próximo token** a partir de esos puntajes, usando una estrategia de selección.

### Dónde empieza nuestro ejercicio

**No implementamos el modelo que calcula los logits.** Los escribimos a mano:

```python
token_logits: dict[str, float] = {
    "azul": 2.0,
    "verde": -1.0,
    ".": 0.0,
    "gris": 3.0
}
```

Son cuatro candidatos de juguete, no el vocabulario completo ni los identificadores reales de un tokenizer. El programa no analiza una frase, no entrena una red y no calcula atención. Empieza con los logits disponibles y realiza cálculos reales para seleccionar un candidato.

La atención está dentro del Transformer y combina información de posiciones del contexto. **Este programa selecciona un token de salida; no selecciona un “token de atención”.**

### Recorrido implementado

```text
Logits escritos a mano
        ↓
División por temperatura
        ↓
Softmax: exponenciar y dividir por la suma
        ↓
Probabilidades originales
        ├── Greedy: mostrar el candidato con probabilidad máxima
        ├── Acumulados: mostrar los límites de los intervalos
        └── Ordenar candidatos por probabilidad
                    ↓
                Filtro top-p
                    ↓
          Renormalizar los elegidos
                    ↓
           Sampling de diez pruebas
```

Greedy se muestra como comparación. No alimenta ni limita el filtro top-p.

---

## 3. Cómo ejecutar e interpretar el resultado

```bash
python3 token_selection.py
```

Solo utiliza módulos de la biblioteca estándar: `math` y `random`. Las anotaciones `list[float]` y `tuple[int, float]` requieren Python 3.9 o posterior.

Con temperatura `1.0` y `top_p = 0.90`, los resultados deterministas esperados son:

| Índice original | Token | Logit | Probabilidad original | Probabilidad después del filtro y reajuste |
|---:|---|---:|---:|---:|
| 0 | azul | 2.0 | 0.2561866396 | 0.2689414214 |
| 1 | verde | -1.0 | 0.0127547817 | 0 |
| 2 | . | 0.0 | 0.0346710914 | 0 |
| 3 | gris | 3.0 | 0.6963874872 | 0.7310585786 |

- Greedy elige `gris`, índice `3`.
- Top-p conserva `gris` y `azul`.
- Sus probabilidades originales suman `0.9525741268224333`.
- La lista final es aproximadamente `[0.2689414, 0.0, 0.0, 0.7310586]`.
- La suma final es aproximadamente `1`.
- Los sorteos finales solo pueden seleccionar `azul` o `gris` con estos datos.

En la ejecución compartida salieron **7 grises y 3 azules**. Otra ejecución puede producir otros conteos. No es una cuota ni un resultado obligatorio.

---

## 4. Función `calculate_probabilities`

```python
def calculate_probabilities(
    logits: list[float],
    temperature: float = 1.0
) -> list[float]:
```

**Propósito:** convertir los logits en probabilidades aplicando temperatura y softmax.

### Entradas y salida

- `logits`: lista de puntajes. Debe mantener el mismo orden que los tokens.
- `temperature`: divisor aplicado a cada logit antes de exponenciar. El valor predeterminado es `1.0`.
- Devuelve `probabilities`: lista de probabilidades en el mismo orden de entrada.
- También imprime cálculos intermedios para aprender; no guarda archivos ni modifica pesos de un modelo.

### Variables internas

| Variable | Significado |
|---|---|
| `exp_values` | Lista de exponenciales de los logits ajustados por temperatura. |
| `logit` | Puntaje que se está procesando en la primera iteración. |
| `exp` | Valor calculado por `math.exp(logit / temperature)`; en el segundo bucle es cada elemento de `exp_values`. |
| `total_exp_values` | Suma de todos los exponenciales. Es el denominador de softmax. |
| `probabilities` | Lista donde se acumula el resultado de dividir cada exponencial por el total. |
| `probability` | Probabilidad individual calculada en esa vuelta. |
| `total_probabilities` | Suma de las probabilidades, impresa como comprobación. |

### Cálculo

Para cada candidato:

```text
logit ajustado = logit / temperature
exponencial = exp(logit ajustado)
probabilidad = exponencial / suma de todos los exponenciales
```

Para `azul`, con temperatura 1:

```text
exp(2.0) = 7.389056...
suma de exponenciales = 28.842472...
7.389056... / 28.842472... = 0.2561866...
```

La función conserva el orden. No ordena candidatos ni elige uno.

### Temperatura

- `0 < T < 1`: concentra más probabilidad en los logits altos.
- `T = 1`: conserva la distribución de softmax sin ajuste.
- `T > 1`: vuelve la distribución más pareja.

**1 es la referencia, no el máximo.** La fórmula admite temperaturas positivas finitas; el código tiene límites numéricos. `T = 0` causa división por cero. Una API puede imponer otro rango o un tratamiento especial de cero.

La temperatura no cambia los pesos del modelo. En este programa tampoco modifica la lista original `logits`: calcula `logit / temperature` dentro de `math.exp`.

Pruebas realizadas con los mismos logits:

| Temperatura | Probabilidad de gris |
|---:|---:|
| 0.5 | 87.86% |
| 1.0 | 69.64% |
| 2.0 | 50.89% |

En las tres pruebas, gris siguió siendo el máximo. Para los mismos logits, una temperatura positiva conserva su orden matemático.

---

## 5. Preparación de las listas y correspondencia por índice

```python
logits = list(token_logits.values())
tokens = list(token_logits.keys())
probabilities = calculate_probabilities(logits, )
```

- `token_logits` mantiene unido el nombre del candidato a su puntaje.
- `logits` extrae los valores: `[2.0, -1.0, 0.0, 3.0]`.
- `tokens` extrae las claves: `["azul", "verde", ".", "gris"]`.
- Ambas extracciones respetan el orden de inserción del mismo diccionario, que no se modifica entre ellas.
- La coma final en `calculate_probabilities(logits, )` es válida. Al no pasar un segundo argumento, se usa temperatura `1.0`.

El índice es la unión entre las listas:

```text
índice 3 → tokens[3] = gris → logits[3] = 3.0
         → probabilities[3] = 0.696387...
```

Cambiar solo los logits, sin mover sus tokens correspondientes, cambia la asignación de puntajes. Reordenar pares completos conserva la asociación.

---

## 6. Bloque greedy

```python
best_index = 0
best_probability = probabilities[0]
```

**Propósito:** encontrar el candidato con mayor probabilidad.

Se toma el primero como mejor candidato provisional. No se asume que sea el máximo. El bucle compara cada probabilidad y actualiza tanto el valor como el índice cuando encuentra una mayor.

| Variable | Significado |
|---|---|
| `best_index` | Índice original del mejor candidato encontrado hasta el momento. |
| `best_probability` | Mayor probabilidad encontrada hasta ese momento. |
| `index` | Índice que entrega `enumerate` en la vuelta actual. |
| `probability` | Probabilidad del candidato actual. |
| `token` | Texto obtenido de `tokens[index]`, usado para mostrarlo. |

La condición `best_probability < probability` usa una comparación estricta. Ante un empate, conserva el primer máximo encontrado.

Greedy no sortea: con estas probabilidades elige `gris` siempre. Que sea el candidato más probable no garantiza que una respuesta sea verdadera.

---

## 7. Función `cumulative_probability`

```python
def cumulative_probability(probabilities: list[float]) -> float:
```

**Propósito:** mostrar cómo crece la suma acumulada y devolver el total final.

| Variable | Significado |
|---|---|
| `probabilities` | Lista recibida como entrada. |
| `probability` | Acumulador local, inicializado en `0.0`. En esta función el nombre representa una suma, no una probabilidad individual. |
| `index` | Índice actual producido por `enumerate`. |
| `prob` | Probabilidad individual que se suma. |

Con la distribución original imprime límites cercanos a:

```text
0.2561866
0.2689414
0.3036125
1.0
```

**No devuelve una lista de acumulados.** Devuelve solo el último total, aproximadamente `1.0`. Es una ayuda de observación; el selector calcula sus propios acumulados.

### Probabilidad individual frente a acumulada

- Individual: cuánto mide el tramo de un token.
- Acumulada: dónde termina ese tramo, contando desde cero.

Por ejemplo, el punto tiene probabilidad individual `0.0346711`, pero su tramo original va de `0.2689414` incluido a `0.3036125` excluido.

---

## 8. Función `select_token_index`

```python
def select_token_index(probabilities: list[float], draw: float) -> int:
```

**Propósito:** elegir un índice según el punto sorteado y los intervalos de probabilidad.

### Entradas

- `probabilities`: distribución que se usará en esa selección. La versión final le pasa `filtered_probabilities`.
- `draw`: número del sorteo, entre `0` incluido y `1` excluido. No es la probabilidad de un token. El nombre significa “sorteo” en este contexto.

### Variables y regla

- `cumulative`: acumulado local que empieza en cero.
- `index`: posición del candidato en la lista recibida.
- `probability`: probabilidad individual de esa posición.

En cada vuelta suma la probabilidad. Devuelve el **primer índice para el cual `draw < cumulative`**. El `return` termina la función.

Con la distribución final:

- Azul ocupa `[0, 0.2689414)`.
- Verde y punto tienen tramos de longitud cero.
- Gris ocupa `[0.2689414, 1)`.

Por eso `draw = 0.20` elige azul y `draw = 0.50` elige gris. No busca el límite más cercano.

La anotación `-> int` expresa la intención, pero no obliga a Python a devolver un entero: si el bucle no encuentra un intervalo, la función termina y devuelve `None`. Ver las limitaciones al final.

---

## 9. Ordenamiento y filtro top-p

### Lista `candidates`

```python
candidates: list[tuple[int, float]] = []
for index, probability in enumerate(probabilities):
    candidates.append((index, probability))
candidates.sort(key=lambda candidate: candidate[1], reverse=True)
```

Cada tupla conserva `(índice original, probabilidad)`.

- `candidate[1]` es la probabilidad usada como criterio de orden.
- `reverse=True` ordena de mayor a menor.
- `sort` modifica la propia lista.
- `candidate` es el parámetro local de la pequeña función `lambda`.

El resultado tiene índices `3, 0, 2, 1`: gris, azul, punto, verde. Mover una tupla no cambia el índice guardado dentro de ella.

### Lista `selected_candidates`

```python
selected_candidates: list[tuple[int, float]] = []
cumulative = 0.0
top_p = 0.90
for index, probability in candidates:
    selected_candidates.append((index, probability))
    cumulative += probability
    if cumulative >= top_p:
        break
    print(index)
```

**Propósito:** conservar el grupo mínimo de candidatos más probables cuya suma alcance o supere el umbral `top_p`.

Orden de las operaciones:

1. Guardar el candidato.
2. Sumar su probabilidad.
3. Detenerse si ya se alcanzó o superó el umbral.

Se incluye al candidato que cruza el umbral. No se busca quedar siempre por debajo de 0.90.

- Gris solo: acumulado `0.696387...`; todavía no alcanza.
- Agregamos azul: acumulado `0.952574...`; alcanza, detenemos el bucle.

`print(index)` está después de `break`, por lo que solo muestra `3`: eso no significa que azul esté ausente de la lista. El `print(selected_candidates)` final es el que muestra la selección completa.

El `cumulative` de este bloque está en el nivel principal del script. Es distinto de las variables locales con el mismo nombre dentro de las funciones. Su valor final se usa después como denominador.

### Diferencia con top-k

- **Top-k:** conserva una cantidad fija de candidatos. Con la lista ordenada, `candidates[:2]` conserva los primeros dos.
- **Top-p:** conserva una cantidad variable, determinada por la suma acumulada.

El corte top-k se probó antes y luego se reemplazó para estudiar top-p por separado. **La versión actual contiene top-p, no una combinación de ambos filtros.** Que k=2 y p=0.90 elijan los mismos candidatos en este ejemplo no los vuelve equivalentes en general.

---

## 10. Renormalización: el cálculo que completa el filtro

```python
filtered_probabilities: list[float] = [0.0] * len(probabilities)
for index, probability in selected_candidates:
    filtered_probabilities[index] = probability / cumulative
```

**Propósito:** repartir el total de probabilidad entre los candidatos conservados, manteniendo sus proporciones.

La suma original de los elegidos es:

```text
cumulative = 0.2561866396 + 0.6963874872 = 0.9525741268
```

La división ocurre en `probability / cumulative`:

```text
azul: 0.2561866396 / 0.9525741268 = 0.2689414214
gris: 0.6963874872 / 0.9525741268 = 0.7310585786
```

No se divide por `top_p` (0.90), sino por **la suma real de los candidatos conservados** (0.952574...). Tampoco se usan como entrada los valores ya reajustados.

### Por qué necesitamos reajustar

Nuestro selector sortea sobre todo el intervalo de 0 a 1. Si dejáramos la suma en 0.952574, una parte de ese intervalo no tendría candidato. Renormalizar hace que la distribución de los elegidos vuelva a sumar 1.

Ejemplo simple repasado:

```text
Quedan 0.40 y 0.10. Su suma es 0.50.
0.40 / 0.50 = 0.80
0.10 / 0.50 = 0.20
```

El primero sigue teniendo cuatro veces la probabilidad del segundo.

### Por qué se crea una lista de cuatro ceros

`[0.0] * len(probabilities)` repite el contenido de una lista. Como hay cuatro probabilidades, produce `[0.0, 0.0, 0.0, 0.0]`.

No es una multiplicación numérica por cero ni una reserva de capacidad. La lista tiene cuatro elementos reales y sigue siendo dinámica:

- `[] * 4` produce `[]`.
- `[4]` es una lista con un elemento, el número 4.
- `[0.0] * 4` tiene cuatro ceros.

Solo recorremos dos candidatos elegidos, pero escribimos en sus **índices originales**: 3 y 0. Las otras posiciones quedan en cero. Así `tokens[selected_index]` sigue identificando correctamente el resultado, sin traducir índices entre listas de distinto tamaño.

---

## 11. Diez sorteos finales

```python
for attempt in range(10):
    draw: float = random.random()
    selected_index = select_token_index(filtered_probabilities, draw)
    print(f"draw: {draw:.4f}, selected_token: {tokens[selected_index]}")
```

Este bloque debe ejecutarse **después** de construir `filtered_probabilities`.

- `attempt`: contador de las diez vueltas; no se usa en el cuerpo.
- `random.random()`: genera un valor pseudoaleatorio uniforme en `[0, 1)`.
- `draw`: posición sorteada en el intervalo completo.
- `selected_index`: índice devuelto por el selector.
- `tokens[selected_index]`: texto del candidato elegido.
- `:.4f`: muestra cuatro decimales. La selección utiliza el valor completo, no el redondeado que se imprime.

Las probabilidades se calculan una sola vez. Estas diez vueltas son pruebas independientes con la misma distribución, **no una frase de diez tokens**. En un LLM real, cada token elegido se agrega al contexto y el modelo vuelve a producir logits para la siguiente generación.

---

## 12. Probabilidad y frecuencia no son lo mismo

Que gris tenga 73.11% de probabilidad no obliga a que salga siete veces en diez pruebas.

- La probabilidad describe las chances de cada sorteo bajo esta distribución.
- La frecuencia observada cuenta lo que ocurrió en una tanda concreta.
- Con muchísimas pruebas independientes, las frecuencias tienden a aproximarse a las probabilidades; no tienen que acercarse de forma continua en cada nueva prueba.
- Con 10% por sorteo, no está garantizado que un candidato aparezca una vez en diez intentos.

En la prueba anterior de 100 sorteos **sin filtro**, los conteos fueron gris 71, azul 23, punto 4 y verde 2. Esa tanda no cambió las probabilidades originales del programa.

---

## 13. Alcance y limitaciones de esta versión educativa

El ejercicio quedó completado para los datos probados. No es una implementación preparada para cualquier entrada ni para producción:

- Softmax calcula exponenciales directamente. No implementa la variante numéricamente estable que resta el máximo antes de exponenciar; logits grandes o temperaturas muy pequeñas pueden provocar desbordamiento.
- No valida temperaturas positivas y finitas, logits válidos o listas vacías.
- Greedy accede a `probabilities[0]`, por lo que una lista vacía produce un error.
- No valida `0 < top_p <= 1`. Los valores fuera del rango no tienen un tratamiento explícito.
- No comprueba que `draw` esté en `[0, 1)` ni que la distribución recibida sea válida.
- `select_token_index` puede devolver `None` si no encuentra intervalo, por entradas inválidas o una diferencia de redondeo en la suma final. La anotación `-> int` no evita ese comportamiento.
- Los `float` pueden producir sumas como `1.0000000000000002`. Es una limitación de representación numérica, no necesariamente un error conceptual.
- El denominador `cumulative` se conserva como variable del script y se reutiliza en otro bloque. El orden de ejecución importa.
- Las anotaciones de tipos documentan la intención; Python no impone por sí solo sus restricciones en ejecución.
- No se fija una semilla aleatoria, por lo que los sorteos pueden cambiar entre ejecuciones.
- El programa no prueba la exactitud factual del texto ni si un token es apropiado para un contexto real.

Estas notas describen límites del código existente; no son tareas adicionales para reabrir este ejercicio.

## 14. Guía rápida para repasar

Antes de mirar las explicaciones, intentar responder:

1. ¿De dónde vienen los logits en este archivo y de dónde vendrían en un modelo real?
2. ¿Qué devuelve softmax? ¿Ya eligió un token?
3. ¿Qué diferencia hay entre greedy y sampling?
4. ¿Temperatura 1 es un máximo o una referencia?
5. ¿Por qué top-p incluye al candidato que cruza el umbral?
6. ¿Por qué el denominador de renormalización es `cumulative` y no `top_p`?
7. ¿Por qué la lista filtrada conserva cuatro posiciones si solo dos candidatos participan?
8. ¿Por qué diez sorteos de este archivo no equivalen a generar una frase?

**Próximo paso acordado:** aplicar y comparar configuraciones de generación con Mistral real. Conservar este archivo como referencia de los mecanismos ya practicados.
