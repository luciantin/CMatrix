import random
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

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



def genTestArray(len, samples):
    res = np.empty((0,len), int)
    for y in range(0, samples):
        ar = np.array([])
        for x in range(0, len):
            ar = np.append(ar, random.randint(0, 1))
        res = np.vstack((res,ar))
    return res

def genTruthArray(truthAr):
    res = np.empty((0,1), int)
    for y, subAr in enumerate(truthAr):
        ar = np.array([])
        # if()
        ar = subAr[0]
        # for x in range(0, ):
        #     ar = np.append(ar, random.randint(-10, 10))
        res = np.vstack((res,ar))
    return res



# testAr = genTestArray(5, 50)
# truthAr = genTruthArray(testAr)

# X, y = spiral_data(samples=10, classes=2)

X, y = spiral_data(samples=1000, classes=2)
_X, _y = spiral_data(samples=100, classes=5)

with open("AutoGenTest.h", "w") as text_file:
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

