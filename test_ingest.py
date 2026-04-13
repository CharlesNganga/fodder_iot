import json
import hmac
import hashlib
import requests
from datetime import datetime, timezone

URL = "http://127.0.0.1:8000/api/ingest/"

# Read secret and strip quotes
secret = ""
with open('.env', 'r') as f:
    for line in f:
        if line.startswith('ESP32_HMAC_SECRET='):
            secret = line.strip().split('=', 1)[1].strip("'\"")
            break

data = {
    "device_id": "TEST:ESP32:001",
    "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    "latitude": -1.1018,
    "longitude": 37.0144,
    "ph": 5.8,
    "ec_raw": 310.2,
    "moisture": 42.1,
    "temperature": 23.5,
    "battery_v": 4.0
}

# Payload must have NO spaces between keys/values
payload_str = json.dumps(data, separators=(',', ':'))
signature = hmac.new(
    secret.encode('utf-8'), 
    payload_str.encode('utf-8'), 
    hashlib.sha256
).hexdigest()

headers = {
    "Content-Type": "application/json",
    "X-Esp32-Signature": signature
}

print(f"🚀 Sending: {payload_str}")
print(f"🔑 Signature: {signature}")

response = requests.post(URL, data=payload_str, headers=headers)
print(f"📡 Status: {response.status_code}")
print(f"📦 Response: {response.json()}")
