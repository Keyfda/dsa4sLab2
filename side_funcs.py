from PIL import Image
import numpy as np
import struct


def merge_matrices(sub_matrices, original_rows, original_cols):
    num_blocks_per_row = original_cols // 8
    num_blocks_per_col = original_rows // 8
    matrix = np.zeros((original_rows, original_cols))
    for block_idx, sub_matrix in enumerate(sub_matrices):
        start_row = (block_idx // num_blocks_per_row) * 8
        start_col = (block_idx % num_blocks_per_row) * 8

        for i, row in enumerate(sub_matrix):
            for j, value in enumerate(row):
                matrix[start_row + i][start_col + j] = value

    return matrix


def make_arr(filename: str, n) -> np.array:
    with Image.open(filename) as img:
        img = img.convert('RGB')

    num_rows = img.height
    num_columns = img.width
    data = img.load()
    arr = np.zeros((num_rows, num_columns, n), dtype=np.uint8)
    for y in range(num_rows):
        for x in range(num_columns):
            for i in range(n):
                arr[y, x, i] = data[x, y][i]
    # print(arr)

    return arr


def diagmatrix(matrix):
    N = min(len(matrix), len(matrix[0]))
    M = max(len(matrix), len(matrix[0]))
    l = []
    for i in range(N + M):
        if i < N and i % 2 == 1:
            j = 0
            for j in range(0, i + 1):
                l.append(matrix[j][i - j])
        elif i < N and i % 2 == 0:
            j = 0
            for j in range(0, i + 1):
                l.append(matrix[i - j][j])
        elif N <= i < M and i % 2 == 1:
            j = 0
            for j in range(0, N):
                l.append(matrix[j][i - j])
        elif N <= i < M and i % 2 == 0:
            j = 0
            for j in range(0, N):
                l.append(matrix[N - j - 1][i - N + j + 1])
        elif i >= M and i % 2 == 1:
            j = 0
            for j in range(i - M + 1, N):
                l.append(matrix[j][i - j])
        elif i >= M and i % 2 == 0:
            j = 0
            for j in range(N - 1, i - M, -1):
                l.append(matrix[j][i - j])
    return (l)


def split_matrix(matrix):
    result = []
    num_rows, num_cols = len(matrix), len(matrix[0])

    for i in range(0, num_rows, 8):
        for j in range(0, num_cols, 8):
            sub_matrix = [row[j:j + 8] for row in matrix[i:i + 8]]
            result.append(sub_matrix)

    return result


def diagchan(layer_array):
    layer_vector = []
    for i in range(len(layer_array)):
        layer_vector.extend(diagmatrix(layer_array[i]))
    # print(min(layer_vector), max(layer_vector))
    return int16_array_to_bytes(layer_vector)


def int16_array_to_bytes(int16_array):
    byte_array = bytearray()
    for num in int16_array:
        packed = struct.pack('h', num)
        byte_array.extend(packed)
    return bytes(byte_array)


def divide_chans(ycbcr_array, width, height):
    y_chan = np.zeros((height, width))
    cb_chan = np.zeros((height, width))
    cr_chan = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            y_chan[i][j] = ycbcr_array[i, j, 0]
            cb_chan[i][j] = ycbcr_array[i, j, 1]
            cr_chan[i][j] = ycbcr_array[i, j, 2]

    return y_chan, cb_chan, cr_chan


def divide_into_blocks(image_array, width, height):
    block_size = 8
    # assert height % block_size == 0
    # assert width % block_size == 0

    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image_array[y:y + block_size, x:x + block_size]
            blocks.append(block)
    blocks = np.array(blocks)

    return blocks


def bytes_to_int16_array(byte_array):
    # Создание массива int16 из массива байтов
    int16_array = []
    for i in range(0, len(byte_array), 2):  # Каждое int16 занимает 2 байта
        # Использование struct.unpack для распаковки байтов в int16
        num = struct.unpack('h', byte_array[i:i+2])[0]
        int16_array.append(num)
    return int16_array


def layer_vector_to_matrix(vector):
    layer_array = list()
    for i in range(0, len(vector), 64):
        work_vector = vector[i:i+64]
        layer_array.append(vector_to_matrix(work_vector))
    return layer_array


def vector_to_matrix(vector):
    matrix = [[0 for _ in range(8)] for _ in range(8)]
    v_index = 0
    for diag in range(15):
        start_row = 0 if diag < 8 else diag - 7
        end_row = diag if diag < 8 else 7
        if diag % 2 == 0:
            for i in range(end_row, start_row - 1, -1):
                matrix[i][diag - i] = vector[v_index]
                v_index += 1
        else:
            for i in range(start_row, end_row + 1):
                matrix[i][diag - i] = vector[v_index]
                v_index += 1
    return matrix
