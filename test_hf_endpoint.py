import os
from dotenv import load_dotenv
load_dotenv()

import requests

token = os.environ.get("HF_TOKEN")
print("Token loaded:", "YES" if token else "NO - missing from .env")

url = "https://router.huggingface.co/hf-inference/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

r = requests.post(
    url,
    headers={"Authorization": f"Bearer {token}"},
    json={"inputs": ["hello world"]}
)

print("Status code:", r.status_code)
print("Response text:", r.text[:1000])