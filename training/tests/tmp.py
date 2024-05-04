import os
import urllib.request
from zipfile import ZipFile

URL = "https://nnfs.io/datasets/fashion_mnist_images.zip"
FILE = "fashion_mnist_images.zip"
FOLDER = "fashion_mnist_images"

if not os.path.isfile(FILE):
    print(f'Downloading {URL} to {FILE}')
    urllib.request.urlretrieve(URL, FILE)

print(f'Unzipping {FILE}')
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)

print("Done")