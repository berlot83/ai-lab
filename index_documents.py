"""Indexación incremental: Qdrant guarda tanto los vectores como el texto original.

Requiere Python 3.10+, Qdrant en localhost:6333 y Ollama en localhost:11434
con el modelo bge-m3 descargado. Ejecutar: python3 index_documents.py
"""

import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
COLLECTION = "documentation"
MODEL = "bge-m3"
# Dimensión verificada para bge-m3 en este laboratorio, no para cualquier modelo.
VECTOR_SIZE = 1024
COLLECTION_URL = f"{QDRANT_URL}/collections/{COLLECTION}"

# El ID identifica al documento: NO lo cambies cuando edites su texto.
# Para agregar otro documento, asignale un ID nuevo que no esté en uso.
DOCUMENTS: list[tuple[int, str]] = [
    (1, "El servicio reintenta hasta 3 veces ante errores transitorios."),
    (2, "El timeout predeterminado es de 30 segundos."),
    (3, "Los errores de autenticación no deben reintentarse."),
    (4, "Ante errores transitorios, el servicio espera 5 segundos entre reintentos."),
]


def request_json(method: str, url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Envía JSON y devuelve la respuesta. Los errores no se ignoran."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def ensure_collection() -> None:
    """Crea la colección solo si falta; nunca borra una existente."""
    try:
        result = request_json("GET", COLLECTION_URL)["result"]
    except HTTPError as error:
        if error.code != 404:
            raise
        request_json("PUT", COLLECTION_URL, {
            "vectors": {"size": VECTOR_SIZE, "distance": "Cosine"},
        })
        print(f"Colección creada: {COLLECTION}")
        return

    vectors = result["config"]["params"]["vectors"]
    if vectors.get("size") != VECTOR_SIZE or vectors.get("distance") != "Cosine":
        raise ValueError(
            "La colección debe tener un vector sin nombre de 1024 dimensiones "
            "con distancia Cosine. No se modificó su configuración."
        )


def retrieve_documents(ids: list[int]) -> dict[int, dict[str, Any]]:
    """Qdrant omite del resultado los IDs que todavía no existen."""
    if not ids:
        return {}
    response = request_json("POST", f"{COLLECTION_URL}/points", {
        "ids": ids,
        "with_payload": True,
        "with_vector": False,
    })
    return {point["id"]: point.get("payload") or {} for point in response["result"]}


def generate_embedding(text: str) -> list[float]:
    response = request_json("POST", f"{OLLAMA_URL}/api/embed", {
        "model": MODEL,
        "input": text,
    })
    vector = response["embeddings"][0]
    if len(vector) != VECTOR_SIZE:
        raise ValueError(f"Se esperaban {VECTOR_SIZE} dimensiones, llegaron {len(vector)}.")
    return vector


def index_documents(documents: list[tuple[int, str]]) -> None:
    if not documents:
        print("No hay documentos para procesar.")
        return

    ids = [doc_id for doc_id, _ in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Hay IDs repetidos en DOCUMENTS; cada documento necesita uno único.")

    ensure_collection()
    # Primero leemos Qdrant. No llamamos a Ollama para textos sin cambios.
    existing = retrieve_documents(ids)
    points: list[dict[str, Any]] = []
    skipped = 0
    for doc_id, text in documents:
        if doc_id in existing and existing[doc_id].get("text") == text:
            skipped += 1
            print(f"ID {doc_id}: sin cambios, omitido.")
            continue
        print(f"ID {doc_id}: generando embedding (nuevo o modificado).")
        # Conservamos otros campos del payload si el punto ya existía.
        payload = {**existing.get(doc_id, {}), "text": text}
        points.append({"id": doc_id, "vector": generate_embedding(text), "payload": payload})

    if points:
        # Upsert inserta IDs nuevos y reemplaza los existentes con el mismo ID.
        request_json("PUT", f"{COLLECTION_URL}/points?wait=true", {"points": points})
    print(f"Listo: {len(points)} insertados/actualizados; {skipped} sin cambios.")


# Supone que los vectores existentes también fueron generados con bge-m3.
# Cambiar el modelo o su configuración exige regenerar TODOS los embeddings,
# aunque el texto siga igual. Este script no detecta ese cambio automáticamente.
# Quitar un documento de DOCUMENTS NO borra su punto de Qdrant.
if __name__ == "__main__":
    index_documents(DOCUMENTS)
