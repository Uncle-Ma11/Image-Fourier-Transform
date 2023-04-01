import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LogNorm
from numpy import savetxt


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
    print(f)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(np.abs(f), norm=LogNorm(), cmap='gray')
    plt.title('Logarithmic Colormap'), plt.xticks([]), plt.yticks([])
    plt.show()


def mode_two(img):
    # f = fft_2d(img)
    f = np.fft.fft2(img)

    denoise_fraction = 0.1
    print('Denoise Fraction: ' + str(denoise_fraction))

    M, N = f.shape
    print('Number of non-zeros: ' + str(int(M * (1 - denoise_fraction * 2))) + 'x' + str(
        int(N * (1 - denoise_fraction * 2))))

    f[int(M * denoise_fraction):int(M * (1 - denoise_fraction))] = 0
    f[:, int(N * denoise_fraction):int(N * (1 - denoise_fraction))] = 0

    # denoised_img = inverse_fft_2d(f).real
    denoised_img = np.fft.ifft2(f).real
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(denoised_img, cmap='gray')
    plt.title('Denoised Version'), plt.xticks([]), plt.yticks([])
    plt.show()


def mode_three(img):
    # f= fft_2d(img)
    f = np.fft.fft2(img)

    file_name = 'original.csv'
    savetxt(file_name, f, delimiter=',')
    print('original size: ' + str(os.path.getsize(file_name)) + ' bytes')

    plt.subplot(231), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    compress_fractions = [20, 40, 60, 80, 95]
    plt_count = 1

    for fraction in compress_fractions:
        plt_count += 1
        trans = f.copy()
        threshold = np.percentile(abs(trans), fraction)
        trans[np.abs(trans) < threshold] = 0
        file_name = str(fraction) + '%.csv'
        savetxt(file_name, trans, delimiter=',')
        print(str(fraction) + '% compressed size: ' + str(os.path.getsize(file_name)) + ' bytes')
        # compressed = inverse_fft_2d(trans).real
        compressed = np.fft.ifft2(trans).real
        plt.subplot(2, 3, plt_count), plt.imshow(compressed, cmap='gray')
        plt.title(str(fraction) + '% compressed'), plt.xticks([]), plt.yticks([])

    plt.suptitle('Compressed images')
    plt.show()


def mode_four(img):
    arr5 = np.random.rand(2 ** 5, 2 ** 5)
    arr6 = np.random.rand(2 ** 6, 2 ** 6)
    arr7 = np.random.rand(2 ** 7, 2 ** 7)
    arr8 = np.random.rand(2 ** 8, 2 ** 8)
    arr9 = np.random.rand(2 ** 9, 2 ** 9)

    test_arrs = [arr5, arr6, arr7, arr8, arr9]
    problem_sizes = ['2^5 x 2^5', '2^6 x 2^6', '2^7 x 2^7', '2^8 x 2^8', '2^9 x 2^9']

    dft_avgs = []
    fft_avgs = []
    dft_ci = []
    fft_ci = []

    power = 5

    for arr in test_arrs:
        dft_times = []
        fft_times = []
        for i in range(1, 10):
            start = time.time()
            naive_2ddft(arr)
            end = time.time()
            dft_times.append(end - start)

            start = time.time()
            fft_2d(arr)
            end = time.time()
            fft_times.append(end - start)
        dft_avg = np.average(dft_times)
        dft_std = np.std(dft_times)
        fft_avg = np.average(fft_times)
        fft_std = np.std(fft_times)
        dft_avgs.append(dft_avg)
        fft_avgs.append(fft_avg)
        dft_ci.append(dft_std * 2)
        fft_ci.append(fft_std * 2)
        print("Naive DFT power of " + str(power) + " average runtime: " + str(dft_avg) + "s")
        print("Naive DFT power of " + str(power) + " runtime standard deviation: " + str(dft_std) + "s")
        print("FFT power of " + str(power) + " average runtime: " + str(fft_avg) + "s")
        print("FFT power of " + str(power) + " runtime standard deviation: " + str(fft_std) + "s")
        power += 1

    plt.figure(figsize=(15, 5))
    plt.title('DFT Runtime Analysis')
    plt.xlabel('problem size')
    plt.ylabel('average runtime(s)')

    plt.errorbar(x=problem_sizes, y=dft_avgs, yerr=dft_ci, label='naive dft')
    plt.errorbar(x=problem_sizes, y=fft_avgs, yerr=fft_ci, label='fft')
    plt.legend()
    plt.show()






if __name__ == "__main__":
    argv = read_input()
    mode = argv.mode
    image = argv.image
    padded_img = read_image(image)

    if mode == 1:
        mode_one(padded_img)
    elif mode == 2:
        mode_two(padded_img)
    elif mode == 3:
        mode_three(padded_img)
    elif mode == 4:
        mode_four(padded_img)
    else:
        print("Invalid Input")

    # test = np.array([[0, 1, 0, 1], [1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]])
    # dft1 = np.fft.ifft2(test)
    # print(dft1)
    # dft2 = inverse_fft_2d(test)
    # print(dft2)
    # print(np.allclose(dft1, dft2))
