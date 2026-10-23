import json
import math
from urllib.request import Request, urlopen
from http.client import HTTPResponse

document_chunks = [
"El servicio reintenta hasta 3 veces ante errores transitorios.",
"El timeout predeterminado es de 30 segundos.",
"Los errores de autenticación no deben reintentarse."
]

question = "¿Cuántos segundos espera el servicio entre reintentos?"

def generate_embedding(text: str) ->  list[float]:
    payload: dict[str, str] = {
        "model": "bge-m3",
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
        vector = generate_embedding(chunk)
        embedded_chunks.append((chunk, vector))
    return embedded_chunks

def index_documents() -> dict:
    with open("embeddings.json", "r", encoding = "utf-8") as file:
        loaded_data = json.load(file)
        if loaded_data["model"] != "bge-m3":
            raise ValueError("Modelo no soportado")

    embedded_chunks = loaded_data["chunks"]

    print(len(embedded_chunks))
    print(len(embedded_chunks[0][1]))

    ## Qdrant
    with urlopen("http://localhost:6333/collections/documentation/exists", timeout=120) as response:
        exists_result = json.loads(response.read())
    collection_exists: bool = exists_result["result"]["exists"]

    if not (collection_exists):
        collection_payload = {
            "vectors": {
                "size": len(embedded_chunks[0][1]),
                "distance": "Cosine"
            }
        }
        collection_body: bytes = json.dumps(collection_payload).encode("utf-8")
        request: Request = Request(url = "http://localhost:6333/collections/documentation", data = collection_body, headers = {"Content-Type": "application/json"}, method = "PUT")
        with (urlopen(request, timeout = 120)) as response:
            result = json.loads(response.read().decode("utf-8"))
            print(result)
    else:
        print("Collection already exists.")

    points = []
    for point_id, (chunk, vector) in enumerate(embedded_chunks, start = 1):
        points.append({
            "id": point_id,
            "vector": vector,
            "payload": {"text": chunk}
        })
    print(len(points))
    print(points[0]["id"], points[0]["payload"], len(points[0]["vector"]))

    points_body: bytes = json.dumps({"points": points}).encode("utf-8")
    points_request: Request = Request(url = "http://localhost:6333/collections/documentation/points?wait=true", data = points_body, headers = {"Content-Type": "application/json"}, method = "PUT")
    with (urlopen(points_request, timeout = 120)) as response:
        points_result = json.loads(response.read())
        print("Carga en Qdrant:", points_result)
    return points_result

def retrieve_chunks(_question: str) -> str:
    question_vector = generate_embedding(_question)
    query_payload = {
        "query": question_vector,
        "limit": 2,
        "with_payload": True
    }
    query_body: bytes = json.dumps(query_payload).encode("utf-8")
    query_request: Request = Request(url = "http://localhost:6333/collections/documentation/points/query", data = query_body, headers = {"Content-Type": "application/json"}, method = "POST")
    with (urlopen(query_request, timeout = 120)) as response:
        query_result = json.loads(response.read())

    retrieved_points = query_result["result"]["points"]
    context_chunks: list[str] = []
    for point in retrieved_points:
        chunk = point["payload"]["text"]
        score = point["score"]
        context_chunks.append(chunk)
        print(f"Qdrant: {score:.4f} | Fragmento: {chunk}")
    return "\n".join(context_chunks)

def generate_answer(instructions: str, context: str, quest: str):
    prompt = f"""{instructions}
    Documentación:
    {context}
    Pregunta:
    {quest}
    """
    print(prompt)

    payload: dict[str, str | bool] = {
        "model": "mistral:latest",
        "prompt": prompt,
        "stream": False
    }

    request_body: bytes = json.dumps(payload).encode("utf-8")
    request: Request = Request(url = "http://localhost:11434/api/generate", data = request_body, headers = {"Content-Type": "application/json"}, method = "POST")
    response: HTTPResponse
    with (urlopen(request, timeout = 120)) as response:
        result = json.loads(response.read().decode("utf-8"))
        print("Respuesta:", result["response"])

# index_documents()
context = retrieve_chunks(question)
generate_answer(instructions = """Respondé la pregunta usando únicamente la documentación de abajo.
    Si la documentación no contiene la respuesta, indicá qué información falta.
    Aplicá cada regla únicamente al tipo de error que menciona. No traslades a una categoría los reintentos permitidos para otra. Respondé brevemente y citá el fragmento que respalda tu respuesta
""", context = context, quest = question)


