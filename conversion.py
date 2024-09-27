from PIL import Image
import numpy as np


def rgb_to_ycbcr(rgb_array: np.array) -> (np.array, Image.Image):
    rgb_array = rgb_array.astype(np.float32)

    r, g, b = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    y = np.clip(np.round(y), 0, 255).astype(np.uint8)
    cb = np.clip(np.round(cb), 0, 255).astype(np.uint8)
    cr = np.clip(np.round(cr), 0, 255).astype(np.uint8)

    ycbcr_array = np.stack((y, cb, cr), axis=-1)

    ycbcr_image = Image.fromarray(ycbcr_array, 'YCbCr')

    with open('ycbcrr_image.raw', 'wb') as f:
        ycbcr_array.tofile(f)

    return ycbcr_array, ycbcr_image


def ycbcr_to_rgb(ycbcr_array: np.array) -> (np.array, Image.Image):
    ycbcr_array = ycbcr_array.astype(np.float32)

    y, cb, cr = ycbcr_array[:, :, 0], ycbcr_array[:, :, 1], ycbcr_array[:, :, 2]

    r = y + 1.402 * (cr - 128)
    g = y - 0.3441 * (cb - 128) - 0.7141 * (cr - 128)
    b = y + 1.772 * (cb - 128)

    r = np.clip(np.round(r), 0, 255).astype(np.uint8)
    g = np.clip(np.round(g), 0, 255).astype(np.uint8)
    b = np.clip(np.round(b), 0, 255).astype(np.uint8)

    rgb_array = np.stack((r, g, b), axis=-1)

    rgb_image = Image.fromarray(rgb_array, 'RGB')

    # print("RGB Array:")
    # print(rgb_array)

    with open('rgb_imageee.raw', 'wb') as f:
        rgb_array.tofile(f)

    return rgb_array, rgb_image
