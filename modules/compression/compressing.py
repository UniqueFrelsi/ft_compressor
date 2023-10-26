from math import log10
import numpy as np
import cv2


class Image(object):

    @classmethod
    def get_image(cls, path: str):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img.shape[2] != 4:
            return img
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    @classmethod
    def set_color_model(cls, img: np.ndarray, from_m="BGR", to_m="BGR"):
        if from_m == to_m:
            return img
        elif from_m == "BGR":
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        else:
            return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    @classmethod
    def get_error(cls, img: np.ndarray, com_img: np.ndarray):
        i1, i2, i3 = cv2.split(img)
        c1, c2, c3 = cv2.split(com_img)
        size = img.shape
        mse1 = np.sum((i1 - c1) ** 2) / (3 * size[0] * size[1])
        mse2 = np.sum((i2 - c2) ** 2) / (3 * size[0] * size[1])
        mse3 = np.sum((i3 - c3) ** 2) / (3 * size[0] * size[1])
        psnr1 = 10 * log10(np.max(i1) ** 2 / mse1)
        psnr2 = 10 * log10(np.max(i2) ** 2 / mse1)
        psnr3 = 10 * log10(np.max(i3) ** 2 / mse1)
        mse = [mse1, mse2, mse3]
        psnr = (psnr1 + psnr2 + psnr3) / 3
        return mse, psnr

    @classmethod
    def get_approx(cls, img: np.ndarray, percent: list, n: int):
        for c in range(3):
            color = img[:, :, c]
            p = percent[c]
            N, M = color.shape
            h = 2 ** n
            p = round((100 - p) * (h ** 2) / 100)
            if p == 0:
                color[:] = 0
            else:
                color_norma = np.abs(color)
                for i in range(0, N, h):
                    for j in range(0, M, h):
                        matrix_n = color_norma[i:(i + h), j:(j + h)]
                        list_max = np.partition(matrix_n, (-1) * p, axis=None)[(-1) * p:]
                        list_del = matrix_n < np.min(list_max)
                        matrix_n = color[i:(i + h), j:(j + h)]
                        matrix_n[list_del] = 0
            img[:, :, c] = color
        return img

    @classmethod
    def save_image(cls, img, path):
        isWritten = cv2.imwrite(path, img)
        if not isWritten:
            index = [i for i, c in enumerate(path) if c == "."]
            is_success, im_buf_arr = cv2.imencode(path[index[-1]:], img)
            im_buf_arr.tofile(path)
