import json
from urllib.request import Request, urlopen
from http.client import HTTPResponse

question = "Cuántos segundos espera el servicio entre reintentos?"

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

context = retrieve_chunks(question)
generate_answer(instructions = """
Respondé usando únicamente la documentación. Verificá que el fragmento citado respalde exactamente el dato solicitado. No confundas timeout de una operación con intervalo entre reintentos, ni cantidad de intentos con duración. Si falta el dato, respondé «La documentación no especifica [dato faltante]», sin estimarlo ni inferirlo a partir de otros valores. Aplicá cada regla solo al tipo de error indicado. Respondé brevemente y citá un fragmento solo si respalda la respuesta.
""", context = context, quest = question)

