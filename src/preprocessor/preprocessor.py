import requests
import zipfile
import os

def download_and_unzip(url, extract_to="."):

    os.makedirs(extract_to, exist_ok=True)
    
    zip_path = os.path.join(extract_to, 'dataset.zip')
    response = requests.get(url, stream=True)
    with open(zip_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(zip_path)
    print(f"I file sono stati estratti in: {extract_to}")