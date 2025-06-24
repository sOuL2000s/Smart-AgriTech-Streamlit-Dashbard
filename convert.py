import base64

with open("agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

with open("firebase_key_b64.txt", "w") as out:
    out.write(encoded)

print("Firebase JSON encoded and saved to firebase_key_b64.txt")
