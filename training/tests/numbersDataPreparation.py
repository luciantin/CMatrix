import os
import sys
import urllib.request
from zipfile import ZipFile
import cv2
import matplotlib.pyplot as plt
import numpy as np

# np.set_printoptions(linewidth=200)
np.set_printoptions(threshold=sys.maxsize)

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

def npArrayToMatrixTxt(array):
    if len(array.shape) == 2:
        cAr = f"{array.shape[1]};{array.shape[0]};"
        for y, subAr in enumerate(array):
            for x, el in enumerate(subAr):
                cAr += f"{el};"
        return cAr
    else :
        cAr = f"1;{array.shape[0]};"
        for y, el in enumerate(array):
            cAr += f"{el};"
        return cAr

URL = "https://nnfs.io/datasets/fashion_mnist_images.zip"
# FILE = "fashion_mnist_images.zip"
FOLDER = "numbers_mnist_images"

# if not os.path.isfile(FILE):
#     print(f'Downloading {URL} to {FILE}')
#     urllib.request.urlretrieve(URL, FILE)
#
# if not os.path.isdir(FOLDER):
#     print(f'Unzipping {FILE}')
#     with ZipFile(FILE) as zip_images:
#         zip_images.extractall(FOLDER)
#
# print("Download Done")

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    # print(os.path.join(path, dataset))
    x = []
    y = []
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            x.append(image)
            y.append(label)
            # cv2.imshow('image', image)
            # cv2.waitKey(0)
            # print(label)
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
np.random.shuffle(keys)
np.random.shuffle(keys)
np.random.shuffle(keys)

X = X[keys]
y = y[keys]

_keys = np.array(range(_X.shape[0]))
np.random.shuffle(_keys)
np.random.shuffle(_keys)
np.random.shuffle(_keys)
np.random.shuffle(_keys)

_X = _X[_keys]
_y = _y[_keys]


X = X[:50000]
y = y[:50000]

_X = _X[:100000]
_y = _y[:100000]

# print(X)
# print("---------------")
# print(X)


# with open("datasets/dataset_numbers_100_train.matrix", "w") as text_file:
#     text_file.write(npArrayToMatrixTxt(X))
#
# with open("datasets/dataset_numbers_100_truth_train.matrix", "w") as text_file:
#     text_file.write(npArrayToMatrixTxt(y))
#
# with open("datasets/dataset_numbers_100_test.matrix", "w") as text_file:
#     text_file.write(npArrayToMatrixTxt(_X))
#
# with open("datasets/dataset_numbers_100_truth_test.matrix", "w") as text_file:
#     text_file.write(npArrayToMatrixTxt(_y))
#
#


#
with open("datasets/dataset_numbers_50_000_train.matrix", "w") as text_file:
    text_file.write(npArrayToMatrixTxt(X))

with open("datasets/dataset_numbers_50_000_truth_train.matrix", "w") as text_file:
    text_file.write(npArrayToMatrixTxt(y))

with open("datasets/dataset_numbers_100_000_test.matrix", "w") as text_file:
    text_file.write(npArrayToMatrixTxt(_X))

with open("datasets/dataset_numbers_100_000_truth_test.matrix", "w") as text_file:
    text_file.write(npArrayToMatrixTxt(_y))



#
# with open("datasets/dataset_numbers_1000_train.matrix", "w") as text_file:
#     text_file.write(npArrayToMatrixTxt(X))
#
# with open("datasets/dataset_numbers_1000_truth_train.matrix", "w") as text_file:
#     text_file.write(npArrayToMatrixTxt(y))
#
# with open("datasets/dataset_numbers_1000_test.matrix", "w") as text_file:
#     text_file.write(npArrayToMatrixTxt(_X))
#
# with open("datasets/dataset_numbers_1000_truth_test.matrix", "w") as text_file:
#     text_file.write(npArrayToMatrixTxt(_y))
#

# with open("AutoGenTest.c", "w") as text_file:
#     text_file.write("#include \"../helpers/nnc_matrix.h\"\n")
#
#     text_file.write("\nNNCIMatrixType GetAutoGenTrainingMatrix(){\n")
#     text_file.write(npArrayToC(X, "zeroOneTest"))
#     text_file.write("return zeroOneTest;\n}\n")
#     text_file.write("\nNNCIMatrixType GetAutoGenTrainingTruthMatrix(){\n")
#     text_file.write(npArrayToC(y, "zeroOneTest"))
#     text_file.write("return zeroOneTest;\n}\n")
#
#     text_file.write("\nNNCIMatrixType GetAutoGenTestMatrix(){\n")
#     text_file.write(npArrayToC(_X, "zeroOneTest"))
#     text_file.write("return zeroOneTest;\n}\n")
#     text_file.write("\nNNCIMatrixType GetAutoGenTestTruthMatrix(){\n")
#     text_file.write(npArrayToC(_y, "zeroOneTest"))
#     text_file.write("return zeroOneTest;\n}\n")
#
# with open("AutoGenTest.h", "w") as text_file:
#     text_file.write("#include \"../helpers/nnc_matrix.h\"\n")
#     text_file.write("\nNNCIMatrixType GetAutoGenTrainingMatrix();\n")
#     text_file.write("\nNNCIMatrixType GetAutoGenTrainingTruthMatrix();\n")
#     text_file.write("\nNNCIMatrixType GetAutoGenTestMatrix();\n")
#     text_file.write("\nNNCIMatrixType GetAutoGenTestTruthMatrix();\n")
