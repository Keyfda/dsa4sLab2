import numpy as np


def dsampling_choose(image, cx, cy, ds_type):
    types = {
        'del': dsampling_del,
        'av': dsampling_avg,
        'near': dsampling_nearest
    }

    return types[ds_type](image, cx, cy)


def dsampling_del(image, cx, cy):
    return image[::cx, ::cy]


def dsampling_avg(image, cx, cy):
    height, width = image.shape
    dsampled = np.zeros((height // cx, width // cy))

    for y in range(0, height, cx):
        for x in range(0, width, cy):
            block = image[y:y + cx, x:x + cy]
            dsampled[y // cx, x // cy] = np.sum(block) / (cx * cy)

    return dsampled


def dsampling_nearest(image, cx, cy):
    height, width = image.shape
    dsampled = np.zeros((height // cx, width // cy))

    for y in range(0, height, cx):
        for x in range(0, width, cy):
            block = image[y:y + cx, x:x + cy]
            nearest_pixel = int(np.round(np.mean(block)))
            dsampled[y // cx, x // cy] = nearest_pixel

    return dsampled


def usampling(image, cx, cy):
    height, width = image.shape
    usampled = np.zeros((height * cx, width * cy))

    for y in range(height):
        for x in range(width):
            usampled[y * cx:(y + 1) * cx, x * cy:(x + 1) * cy] = image[y, x]

    return usampled
