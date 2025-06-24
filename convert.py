# convert.py
import base64
with open("agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

with open("firebase_key_b64.txt", "w") as f:
    f.write(encoded)

print("Done. Copy firebase_key_b64.txt content into FIREBASE_KEY_B64 env var on Render.")
