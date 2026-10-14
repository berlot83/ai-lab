import json
import urllib.request
from http.client import HTTPResponse

document_chunks = ["El servicio reintenta hasta 5 veces ante errores transitorios.",
                   "El timeout predeterminado es de 30 segundos.", 
                   "Los errores de autenticación no deben reintentarse."]

search_term = input("Término de búsqueda: ")

def retrieve_chunks(documents: list[str], query: str):
    normalized_query = query.strip()
    if (normalized_query == ""):
        return []
    chunks = []
    for doc in documents:
        if (normalized_query.lower() in doc.lower()):
            chunks.append(doc)
    return chunks

retrieved_chunks = retrieve_chunks(document_chunks, search_term)

if len(retrieved_chunks) > 0:
    question = input("Escribí tu pregunta: ")
    context =  "\n".join(retrieved_chunks)
    prompt = f"""Instrucciones: Respondé usando la documentación, si la documentación no contiene la respuesta, indicá qué información falta

Documentación: 
{context} 

Pregunta: 
{question}
"""
    payload: dict[str, str | bool] = {
        "model": "mistral:latest",
        "prompt": prompt,
        "stream": False
    }
    print(f"Contexto enviado: {context}")
    payload_text =  json.dumps(payload, ensure_ascii = False, indent = 2)
    request_body = payload_text.encode("utf-8")
    request = urllib.request.Request(url = "http://localhost:11434/api/generate", data = request_body, headers= {"Content-Type": "application/json"}, method= "POST" )
    response: HTTPResponse
    with urllib.request.urlopen(request, timeout = 120) as response:
        response_bytes: bytes = response.read()
        response_text = response_bytes.decode()
        result = json.loads(response_text)
    print(result["response"])
else:
    print("No se encontraron fragmentos relevantes")
    
