import json
import urllib.request

document_chunks = ["El servicio reintenta hasta 3 veces ante errores transitorios.",
                   "El timeout predeterminado es de 30 segundos.", 
                   "Los errores de autenticación no deben reintentarse."]

search_term = "errores"
retrieved_chunks = []

def retrieve_chunks(documents, query):
    chunks = []
    for doc in documents:
        if (query.lower() in doc.lower()):
            chunks.append(doc)
    return chunks

retrieved_chunks = retrieve_chunks(document_chunks, search_term)
context = ""

if len(retrieved_chunks) > 0:
    question = "¿Cuántos segundos espera entre reintentos?"
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
    payload_text =  json.dumps(payload, ensure_ascii = False, indent = 2)
    request_body = payload_text.encode("utf-8")
    request = urllib.request.Request(url = "http://localhost:11434/api/generate", data = request_body, headers= {"Content-Type": "application/json"}, method= "POST" )
    print(payload_text)
    print(prompt)
else:
    print("No se encontraron fragmentos relevantes")
    
