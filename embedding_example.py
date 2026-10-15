import json
from urllib.request import Request, urlopen
from http.client import HTTPResponse

payload: dict[str, str] = {
    "model": "nomic-embed-text",
    "input": "El timeout predeterminado es de 30 segundos."
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
print("Dimensiones:", len(vector))
print("Primeros 5 valores:", vector[:5])