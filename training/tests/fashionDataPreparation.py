import os
import urllib.request
from zipfile import ZipFile
import cv2
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(linewidth=200)

def npArrayToC(array, name):
    if len(array.shape) == 2:
        cAr = f"NNCIMatrixType {name} = NNCMatrixAlloc({array.shape[1]}, {array.shape[0]});\n"
        for y, subAr in enumerate(array):
            for x, el in enumerate(subAr):
                cAr += f"{name}->matrix[{y}][{x}] = {el};"
            cAr += "\n"
        return cAr
    else :
        cAr = f"NNCIMatrixType {name} = NNCMatrixAlloc(1, {array.shape[0]});\n"
        for y, el in enumerate(array):
            cAr += f"{name}->matrix[{y}][0] = {el};"
            cAr += "\n"
        return cAr


URL = "https://nnfs.io/datasets/fashion_mnist_images.zip"
FILE = "fashion_mnist_images.zip"
FOLDER = "fashion_mnist_images"

if not os.path.isfile(FILE):
    print(f'Downloading {URL} to {FILE}')
    urllib.request.urlretrieve(URL, FILE)

if not os.path.isdir(FOLDER):
    print(f'Unzipping {FILE}')
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)

print("Download Done")

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    x = []
    y = []
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            x.append(image)
            y.append(label)
    return np.array(x), np.array(y).astype('uint8')

def create_mnist_dataset(path):
    X, y = load_mnist_dataset('train', path)
    _X, _y = load_mnist_dataset('test', path)
    return X, y, _X, _y


X, y, _X, _y = create_mnist_dataset(FOLDER)

X = (X.astype(np.float32) - 127.5) / 127.5
_X = (_X.astype(np.float32) - 127.5) / 127.5

X = X.reshape(X.shape[0], -1)
_X = _X.reshape(_X.shape[0], -1)

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)

X = X[keys]
y = y[keys]

X = X[:2000]
y = y[:2000]

_X = _X[:100]
_y = _y[:100]

with open("AutoGenTest.c", "w") as text_file:
    text_file.write("#include \"helpers/nnc_matrix.h\"\n")

    text_file.write("\nNNCIMatrixType GetAutoGenTrainingMatrix(){\n")
    text_file.write(npArrayToC(X, "zeroOneTest"))
    text_file.write("return zeroOneTest;\n}\n")
    text_file.write("\nNNCIMatrixType GetAutoGenTrainingTruthMatrix(){\n")
    text_file.write(npArrayToC(y, "zeroOneTest"))
    text_file.write("return zeroOneTest;\n}\n")

    text_file.write("\nNNCIMatrixType GetAutoGenTestMatrix(){\n")
    text_file.write(npArrayToC(_X, "zeroOneTest"))
    text_file.write("return zeroOneTest;\n}\n")
    text_file.write("\nNNCIMatrixType GetAutoGenTestTruthMatrix(){\n")
    text_file.write(npArrayToC(_y, "zeroOneTest"))
    text_file.write("return zeroOneTest;\n}\n")

with open("AutoGenTest.h", "w") as text_file:
    text_file.write("#include \"helpers/nnc_matrix.h\"\n")
    text_file.write("\nNNCIMatrixType GetAutoGenTrainingMatrix();\n")
    text_file.write("\nNNCIMatrixType GetAutoGenTrainingTruthMatrix();\n")
    text_file.write("\nNNCIMatrixType GetAutoGenTestMatrix();\n")
    text_file.write("\nNNCIMatrixType GetAutoGenTestTruthMatrix();\n")
