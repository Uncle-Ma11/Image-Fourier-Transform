import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LogNorm


def read_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, dest="mode", default="1", help="choose mode")
    parser.add_argument("-i", type=str, dest="image", default="moonlanding.png",
                        help="choose the image the be processed")

    parsed_args = parser.parse_args()
    return parsed_args


def read_image(f_name):
    img = cv2.imread(f_name, cv2.IMREAD_GRAYSCALE).astype(float)
    height = img.shape[0]
    width = img.shape[1]
    padded_h = np.power(2, np.floor(np.log2(height)) + 1).astype(int)
    padded_w = np.power(2, np.floor(np.log2(width)) + 1).astype(int)

    img = cv2.resize(img, (padded_w, padded_h))

    return img


def naive_dft(array):
    N = len(array)
    dft_output = np.zeros(N, dtype=np.complex_)
    for k in range(N):
        for n in range(N):
            dft_output[k] += array[n] * np.exp(-2j * np.pi * k * n / N)
    return dft_output


def naive_inverse_dft(array):
    N = len(array)
    inverse_dft_output = np.zeros(N, dtype=np.complex_)
    for k in range(N):
        for n in range(N):
            inverse_dft_output[k] += array[n] * np.exp(2j * np.pi * k * n / N)
    return inverse_dft_output / N


def naive_2ddft(array):
    M, N = array.shape
    dft_output = np.zeros((M, N), dtype=np.complex_)

    for m in range(M):
        dft_output[m, :] = naive_dft(array[m, :])

    for n in range(N):
        dft_output[:, n] = naive_dft(dft_output[:, n])

    return dft_output


def naive_inverse_2ddft(array):
    M, N = array.shape
    inverse_dft_output = np.zeros((M, N), dtype=np.complex_)

    for m in range(M):
        inverse_dft_output[m, :] = naive_inverse_dft(array[m, :])

    for n in range(N):
        inverse_dft_output[:, n] = naive_inverse_dft(inverse_dft_output[:, n])

    return inverse_dft_output


def fft(array):
    N = len(array)

    if N == 1:
        return array

    even_part = fft(array[::2])
    odd_part = fft(array[1::2])

    factor = np.exp(-2j * np.pi * np.arange(N) / N)

    sum = np.concatenate([even_part + factor[:int(N / 2)] * odd_part, even_part + factor[int(N / 2):] * odd_part])
    return sum


def inverse_fft(array):
    N = len(array)

    if N == 1:
        return array

    even_part = fft(array[::2])
    odd_part = fft(array[1::2])

    factor = np.exp(2j * np.pi * np.arange(N) / N)

    sum = np.concatenate([even_part + factor[:int(N / 2)] * odd_part, even_part + factor[int(N / 2):] * odd_part])
    return sum / N


def fft_2d(array):
    M, N = array.shape
    dft_output = np.zeros((M, N), dtype=np.complex_)

    for m in range(M):
        dft_output[m, :] = fft(array[m, :])

    for n in range(N):
        dft_output[:, n] = fft(dft_output[:, n])

    return dft_output


def inverse_fft_2d(array):
    M, N = array.shape
    inverse_dft_output = np.zeros((M, N), dtype=np.complex_)

    for m in range(M):
        inverse_dft_output[m, :] = inverse_fft(array[m, :])

    for n in range(N):
        inverse_dft_output[:, n] = inverse_fft(inverse_dft_output[:, n])

    return inverse_dft_output


def mode_one(img):
    f = fft_2d(img)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(np.abs(f), norm=LogNorm(), cmap='gray')
    plt.title('Logarithmic Colormap'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    argv = read_input()
    mode = argv.mode
    image = argv.image
    padded_img = read_image(image)

    if mode == 1:
        mode_one(padded_img)

    elif mode == 2:
        print("")
    elif mode == 3:
        print("")
    elif mode == 4:
        print("")
    else:
        print("")


    # test = np.array([[0, 1, 0, 1], [1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]])
    # dft1 = np.fft.ifft2(test)
    # print(dft1)
    # dft2 = inverse_fft_2d(test)
    # print(dft2)
    # print(np.allclose(dft1, dft2))
