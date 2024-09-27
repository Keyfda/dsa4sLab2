import numpy as np


def add(array, val):
    array_new = np.zeros((len(array), 8, 8))
    for i in range(len(array)):
        for k in range(8):
            for l in range(8):
                array_new[i][k][l] = (array[i][k][l] + val)
    return array_new.astype(np.uint16)


def subtract(array, val):
    array_new = np.zeros((len(array), 8, 8))
    for i in range(len(array)):
        for k in range(8):
            for l in range(8):
                array_new[i][k][l] = (array[i][k][l] - val)
    return array_new.astype(np.int8)


def dct_coeffs(n):
    coeffs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                coeffs[i][j] = 1 / np.sqrt(n)
            else:
                coeffs[i][j] = np.sqrt(2 / n) * np.cos((2 * j + 1) * i * np.pi / (2 * n))
    return coeffs


def dct(coeffs, layers):
    dct_done = []
    for layer in layers:
        a = np.dot(coeffs, layer)
        b = np.dot(a, np.transpose(coeffs))
        dct_done.append(b.astype(np.int16))
    return dct_done


def dct_back(coeffs, layers):
    dct_rev = []
    max_values = []

    for layer in layers:
        a = np.dot(np.transpose(coeffs), layer)
        b = np.dot(a, coeffs)
        back_matrix = b.astype(np.int16)

        dct_rev.append(back_matrix)
        max_values.append(np.max(back_matrix))

    print('MMMAX= ', max(max_values))

    return dct_rev


def quant_matrix(q):
    base = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    if q < 1:
        q = 1
    elif q > 100:
        q = 100

    if q < 50:
        scale = 5000 / q
    else:
        scale = 200 - q * 2

    quantization_matrix = np.floor((base * scale + 50) / 100)
    quantization_matrix[quantization_matrix < 1] = 1

    return quantization_matrix.astype(np.int16)


def quantize(matrix_array, quan_matrix):
    quant_matrix = []
    for matrix in matrix_array:
        res = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                res[i][j] = np.round(matrix[i][j] / quan_matrix[i][j])
                # if res[i][j] > 127:
                #     res[i][j] = 127
                # elif res[i][j] < -128:
                #     res[i][j] = -128
        quant_matrix.append(res.astype(np.int16))
    return quant_matrix


def dequantize(matrix_array, quan_matrix):
    dequant_matrix = []
    for matrix in matrix_array:
        dequant_matrix.append(np.round(matrix * quan_matrix).astype(np.int16))
    return dequant_matrix

