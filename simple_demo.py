import os, json, hmac, hashlib, requests
from datetime import datetime, timezone

# 1. Load HMAC Secret from .env
SECRET = ""
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.startswith("ESP32_HMAC_SECRET="):
                SECRET = line.strip().split("=")[1].strip("'\"")

URL = "http://127.0.0.1:8000/api/ingest/"

print("🌱 Fodder IoT - Manual Sensor Simulator")
print("Leave an input blank and press Enter to use the default value.\n")

# 2. Interactive Loop
while True:
    try:
        ph_in = input("Enter pH [default 6.2] (or 'q' to quit): ")
        if ph_in.lower() == 'q': break
        
        payload = {
            "device_id": "ESP32:MANUAL:TEST",
            "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            "latitude": -1.1018,   # Fixed to Thika Farm boundary
            "longitude": 37.0144,
            "ph": float(ph_in or 6.2),
            "ec_raw": float(input("Enter EC raw [default 310]: ") or 310.0),
            "moisture": float(input("Enter Moisture % [default 50]: ") or 50.0),
            "temperature": float(input("Enter Temp °C [default 23.5]: ") or 23.5),
            "battery_v": 4.1
        }

        # 3. Sign and Send
        body = json.dumps(payload, separators=(',', ':'))
        sig = hmac.new(SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
        
        print("\n🚀 Sending payload to server...")
        r = requests.post(URL, data=body, headers={"Content-Type": "application/json", "X-Esp32-Signature": sig})
        
        print(f"✅ HTTP {r.status_code}: {r.json()}")
        print("👉 Look at Terminal 2 (Celery) to see the XGBoost AI and Rules Engine output!\n" + "─"*50 + "\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")