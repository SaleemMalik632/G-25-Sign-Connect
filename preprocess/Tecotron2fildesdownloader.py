import os
import requests

def download_file(url, dest):
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# URLs of the files to download
files_to_download = {
    'model.py': 'https://raw.githubusercontent.com/NVIDIA/mellotron/master/model.py',
    'hparams.py': 'https://raw.githubusercontent.com/NVIDIA/mellotron/master/hparams.py',
    'layers.py': 'https://raw.githubusercontent.com/NVIDIA/mellotron/master/layers.py',
    'data_utils.py': 'https://raw.githubusercontent.com/NVIDIA/mellotron/master/data_utils.py',
    # 'text.py': 'https://raw.githubusercontent.com/NVIDIA/mellotron/master/text.py'
}

# Download each file
for filename, url in files_to_download.items():
    download_file(url, filename)

print("All files downloaded successfully.")