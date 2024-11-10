import requests
import zipfile
import os
import random
from collections import defaultdict

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

def get_balanced_indices(dataset, num_samples):
    # Raggruppiamo gli indici per classe
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)

    # Calcoliamo quanti campioni per classe vogliamo
    num_classes = len(class_indices)
    samples_per_class = num_samples // num_classes

    # Selezioniamo in modo bilanciato
    balanced_indices = []
    for class_idx in class_indices:
        indices = class_indices[class_idx]
        random.shuffle(indices)
        balanced_indices.extend(indices[:samples_per_class])

    # Shuffle finale degli indici bilanciati
    random.shuffle(balanced_indices)
    return balanced_indices
