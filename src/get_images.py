import os
import shutil
import zipfile

from huggingface_hub import hf_hub_download

BASE_DIR = os.getenv("BASE_DIR")
dir_data = os.path.join(BASE_DIR, "data/raw_data")

if not os.path.exists(dir_data):
    zip_path = hf_hub_download(
        repo_id="gaabsi/pokemon_nn", filename="pokemon.zip", repo_type="dataset"
    )

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dir_data)
