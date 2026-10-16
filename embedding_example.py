import json
import math
from urllib.request import Request, urlopen
from http.client import HTTPResponse

document_chunks = [
"The service retries up to 3 times on transient errors.",
"The default timeout is 30 seconds.",
"Authentication errors must not be retried."
]

question = "What is the default waiting time?"

def generate_embedding(text: str) ->  list[float]:
    payload: dict[str, str] = {
        "model": "nomic-embed-text",
        "input": text
    }
    json_payload: str = json.dumps(payload)
    payload_bytes: bytes = json_payload.encode("utf-8")
    request: Request = Request(data = payload_bytes, url = "http://localhost:11434/api/embed", headers= {"Content-Type": "application/json"}, method= "POST")
    response: HTTPResponse
    with urlopen(request, timeout = 120) as response:
        response_bytes: bytes = response.read()
        response_text = response_bytes.decode()
        result = json.loads(response_text)
    vector: list[float] = result["embeddings"][0]
    return vector

def vector_norm(vector: list[float]) -> float:
    return math.sqrt(dot_product(vector, vector))

def dot_product(a: list[float], b: list[float]) -> float:
    if (len(a) != len(b)):
        raise ValueError("Los vectores deben tener la misma dimensión")
    result: float = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def cosine_similarity(a: list[float], b: list[float]) -> float:
    product = dot_product(a, b)
    norm_a: float = vector_norm(a)
    norm_b: float = vector_norm(b)
    if (norm_a == 0.0 or norm_b == 0.0):
        raise ValueError("La similitud coseno no está definida para vectores de norma cero")
    return product / (norm_a * norm_b)

def embeddings(chunks: list[str]) -> list[tuple[str, list[float]]]:
    embedded_chunks: list[tuple[str, list[float]]] = []
    for chunk in chunks:
        vector = generate_embedding("search_document: " + chunk)
        embedded_chunks.append((chunk, vector))
    return embedded_chunks

embedded_chunks = embeddings(document_chunks)

question_vector = generate_embedding("search_query: " + question)

scored_chunks: list[tuple[str, float]] = []
for (chunk, vector) in embedded_chunks:
    v: float = cosine_similarity(question_vector, vector)
    scored_chunks.append((chunk, v))
scored_chunks.sort(key=lambda item: item[1], reverse=True)

best_chunk, best_score = scored_chunks[0]
print(f"Fragmento recuperado: {best_chunk}")