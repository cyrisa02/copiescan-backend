import base64
import requests

with open("test_facture.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

response = requests.post(
    "http://localhost:8000/extract",
    json={
        "image_base64": f"data:image/jpeg;base64,{encoded_string}",
        "webhook_url": "https://webhook.site/0cea04e1-7559-4195-9f3a-416bba16214e"  # <-- Remplace par ton URL webhook.site
    }
)

print("Status:", response.status_code)
print("Résultat:", response.json())