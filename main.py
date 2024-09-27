from PIL import Image
import numpy as np
import conversion as conv
import side_funcs as sf
import downsampling as ds
import DCT_funcs as dct


def open_img(filename: str) -> np.array:
    with Image.open(filename) as img:
        img = img.convert('RGB')
        image_array = np.array(img, dtype=np.uint8)
    return image_array


rgb_image_array = open_img("png_image.png")

ycbcr_array, ycbcr_image = conv.rgb_to_ycbcr(rgb_image_array)
converted_back_rgb_array, converted_back_rgb_image = conv.ycbcr_to_rgb(ycbcr_array)


arr = sf.make_arr("png_image.png", 3)
# sf.matrix_traversal(arr)


im = Image.open("png_image.png")
width, height = im.size

dct_c = dct.dct_coeffs(8)
quant_c = dct.quant_matrix(3000)

y_chan, cb_chan, cr_chan = sf.divide_chans(ycbcr_array, width, height)


cb_ds, cr_ds = (ds.dsampling_choose(cb_chan, 2, 2, 'av'),
                ds.dsampling_choose(cr_chan, 2, 2, 'av'))


y_div, cb_div, cr_div = (sf.split_matrix(y_chan),
                         sf.split_matrix(cb_ds),
                         sf.split_matrix(cr_ds))


y_shift, cb_shift, cr_shift = (dct.subtract(y_div, 128),
                               dct.subtract(cb_div, 128),
                               dct.subtract(cr_div, 128))

y_dct, cb_dct, cr_dct = (dct.dct(dct_c, y_shift),
                         dct.dct(dct_c, cb_shift),
                         dct.dct(dct_c, cr_shift),)


y_quant, cb_quant, cr_quant = (dct.quantize(y_dct, quant_c),
                               dct.quantize(cb_dct, quant_c),
                               dct.quantize(cr_dct, quant_c))


y_vector, cb_vector, cr_vector = sf.diagchan(y_quant), sf.diagchan(cb_quant), sf.diagchan(cr_quant)

# print(min(y_vector), min(cb_vector), min(cr_vector))

with open('testencwds50.raw', 'wb') as f:
    f.write(y_vector)
    f.write(cb_vector)
    f.write(cr_vector)


def read_chans(data):
    y = list()
    cb = list()
    cr = list()
    for i in range(len(data)):
        if i < len(data)/1.5:
            y.append(int(data[i]))
        elif len(data)/1.5 <= i < len(data)/1.2:
            cb.append(data[i])
        else:
            cr.append(data[i])
    return y, cb, cr


with open('testencwds50.raw', 'rb') as f:
    data = f.read()


data2 = sf.bytes_to_int16_array(data)
y, cb, cr = read_chans(data2)


y_quan, cb_quan, cr_quan = (sf.layer_vector_to_matrix(y),
                            sf.layer_vector_to_matrix(cb),
                            sf.layer_vector_to_matrix(cr))

y_dct, cb_dct, cr_dct = (dct.dequantize(y_quan, quant_c),
                         dct.dequantize(cb_quan, quant_c),
                         dct.dequantize(cr_quan, quant_c))

y_array, cb_array, cr_array = (dct.dct_back(dct_c, y_dct),
                               dct.dct_back(dct_c, cb_dct),
                               dct.dct_back(dct_c, cr_dct))

y_uint, cb_uint, cr_uint = (dct.add(y_array, 128),
                            dct.add(cb_array, 128),
                            dct.add(cr_array, 128))

y_layer, cb_downsampled, cr_downsampled = (sf.merge_matrices(y_uint, height, width),
                                           sf.merge_matrices(cb_uint, int(height/2), int(width/2)),
                                           sf.merge_matrices(cr_uint, int(height/2), int(width/2)))

cb_layer, cr_layer = ds.usampling(cb_downsampled, 2, 2), ds.usampling(cr_downsampled, 2, 2)


def make_chan(layer):
    n = len(layer)
    m = len(layer[0])
    chan = np.zeros(n * m)
    for i in range(n):
        for j in range(m):
            chan[i * m + j] = layer[i][j]
    return chan.astype(np.uint8)


y_chan, cb_chan, cr_chan = make_chan(y_layer), make_chan(cb_layer), make_chan(cr_layer)


def convert_to_rgb(ycbcr_array):
    rgb_array = [[0 for _ in range(3)] for _ in range(len(ycbcr_array))]
    for i in range(0, len(ycbcr_array)):
        rgb_array[i][0] = min(max(0, int(np.round(ycbcr_array[i][0] + 1.402 * (ycbcr_array[i][2] - 128)))), 255)
        rgb_array[i][1] = min(max(0, int(np.round(ycbcr_array[i][0] - (0.114 * 1.772 * (ycbcr_array[i][1] - 128)
                                                                    + 0.229 * 1.402 * (ycbcr_array[i][2] - 128))/0.587))), 255)
        rgb_array[i][2] = min(max(0, int(np.round(ycbcr_array[i][0] + 1.772 * (ycbcr_array[i][1] - 128)))), 255)
    return rgb_array


ycbcr = list(zip(y_chan, cb_chan, cr_chan))
RGB_array = convert_to_rgb(ycbcr)


def create_image_from_rgb_array(rgb_array, width, height):
    image = Image.new('RGB', (width, height))

    for i, pixel in enumerate(rgb_array):
        x = i % width
        y = i // width
        image.putpixel((x, y), tuple(pixel))  #

    return image


image = create_image_from_rgb_array(RGB_array, width, height)
image.save('testdecwds50.png')
