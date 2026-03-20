import requests

url = "https://workwithus.lucioai.com"

headers = {
    "User-Agent": "hari_seldon",
    "Authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoiQWRpdHlhIFBhbmRleSIsImVtYWlsIjoiYXA1MzEyQG55dS5lZHUiLCJkYXRlIjoiMjAyNi0wMy0yMCAxMDo1MTowMyJ9._RQOxrhdJHdpy9eguY42xgx13hiLDogS1J78vI1CaZQ",
    "Content-Type": "application/json",
    "Accept": "*/*",
    "Cache-Control": "no-cache",
}

cookies = {
    "auth_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoiQWRpdHlhIFBhbmRleSIsImVtYWlsIjoiYXA1MzEyQG55dS5lZHUiLCJkYXRlIjoiMjAyNi0wMy0yMCAxMDo1MTowMyJ9._RQOxrhdJHdpy9eguY42xgx13hiLDogS1J78vI1CaZQ",
}

payload = {
    "token": "eyJhbnN3ZXJzIjpbNTAsMTgsMTk2XX0.ab1TcA.aL2YkSvG-q6EjcfKp40M2aibi4M",
    "answers": [50, "Jupiter", 8],
}

response = requests.post(url, json=payload, headers=headers, cookies=cookies)

print(f"Status: {response.status_code}")
print(f"Headers: {dict(response.headers)}")
print(f"Body: {response.text}")
