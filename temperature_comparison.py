
# Objetivo: comparar respuestas de Mistral al mismo prompt con dos temperaturas distintas. Queremos observar el efecto de la configuración, no medir cuál respuesta es verdadera.
# Checklist:
# ☐ 1. Creá un archivo nuevo en ai-lab llamado temperature_comparison.py. No modifiques los archivos del RAG.
# ☐ 2. Definí un mismo prompt, por ejemplo: «Proponé tres nombres breves para una cafetería». Definí también una lista con dos temperaturas: 0.3 y 1.0.
# ☐ 3. Para cada temperatura, enviá una solicitud a la API local de Ollama, en http://localhost:11434/api/generate. Usá el modelo mistral:latest, el mismo prompt y stream desactivado. La temperatura va dentro del objeto options.
# ☐ 4. Leé la respuesta JSON y mostrá la temperatura y el texto generado. Para esta primera prueba, hacé una llamada por temperatura.
# ☐ 5. Ejecutá el archivo y compartime la salida. Vamos a comparar los resultados teniendo presente que dos respuestas solas no alcanzan para sacar una conclusión estadística.
# Cómo reconocer que funciona: al terminar deberías ver dos respuestas, cada una identificada con su temperatura. No necesitás calcular logits, softmax ni sampling: de eso se encarga la generación con Ollama y Mistral.
# Podés reutilizar el patrón de solicitud HTTP y JSON que ya usaste en tus otros archivos.

# Siguiente objetivo: repetir esta misma llamada con 0.3 y 1.0, manteniendo question igual. Los pasos son:
# 1. Crear la lista de temperaturas.
# 2. En cada vuelta, poner la temperatura actual en payload dentro de options, con la clave "temperature".
# 3. Volver a convertir el payload a JSON y bytes en cada vuelta, antes de crear Request. Si lo convertís una sola vez antes del bucle, los cambios posteriores al diccionario no se reflejarán en el cuerpo ya serializado.
# 4. Enviar la solicitud y mostrar juntos la temperatura y response_body["response"].
# La primera llamada ya funciona; ahora cambiamos la configuración entre llamadas. Vos seguí y yo reviso los cambios guardados.

import json
from urllib.request import Request
from urllib.request import urlopen
from http.client import HTTPResponse
import random


def inference(question: str, temperature: float = 1.0) -> str:
    payload: dict[str, str | bool | object] =  {
        "model": "mistral:latest",
        "prompt": question,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    payload_body: bytes = json.dumps(payload).encode("utf-8")
    request: Request = Request(url = "http://localhost:11434/api/generate", data = payload_body, headers = {"Content-Type": "application/json"}, method = "POST")

    response: HTTPResponse
    with urlopen(request, timeout = 120 ) as response:
        response_bytes: bytes = response.read()
        response_body: dict[str, object] = json.loads(response_bytes)
        result: str = response_body["response"]
        return result

q: str = "Proponé tres nombres breves para una cafetería"
for cycle in range(10):
    rand = random.random()
    result: str = inference(question = q, temperature = rand)
    print(f"Temperature: {rand}, Response: {result}")