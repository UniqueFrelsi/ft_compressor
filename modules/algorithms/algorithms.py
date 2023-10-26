from abc import ABC, abstractmethod
import numpy as np


def resize(matrix: np.ndarray, p: int):
    n_0, m_0, k_0 = matrix.shape
    n, m, k_0 = matrix.shape

    if n & (n - 1):
        n = 2 ** n.bit_length()
    if m & (m - 1):
        m = 2 ** m.bit_length()

    if 2 ** p > n:
        n = 2 ** p
    if 2 ** p > m:
        m = 2 ** p

    matrix = np.concatenate((matrix, np.zeros((n - n_0, m_0, k_0))), axis=0)
    matrix = np.concatenate((matrix, np.zeros((n, m - m_0, k_0))), axis=1)
    return matrix * 1.0


def del_zeros(matrix: np.ndarray, out_size: tuple):
    in_size = matrix.shape
    for ax in range(0, 2):
        del_list = [i for i in range(in_size[ax] - 1, out_size[ax] - 1, -1)]
        matrix = np.delete(matrix, del_list, axis=ax)
    return matrix


class FastAlgorithm(ABC):

    @abstractmethod
    def t_row(self, matrix: np.ndarray, n: int, h: int, hf: int):
        pass

    @abstractmethod
    def t_col(self, matrix: np.ndarray, n: int, h: int, hf: int):
        pass

    @abstractmethod
    def it_row(self, matrix: np.ndarray, n: int, h: int, hf: int):
        pass

    @abstractmethod
    def it_col(self, coef: np.ndarray, n: int, h: int, hf: int):
        pass

    @abstractmethod
    def t_2d(self, matrix: np.ndarray, p: int):
        matrix = np.copy(matrix)
        matrix = resize(matrix, p)
        size = matrix.shape
        n = 2 ** p
        h = 2

        while h <= n:
            hf = h // 2
            for j in range(0, size[1], n):
                matrix[:, j:(j + n), :] = self.t_row(matrix[:, j:(j + n), :], n, h, hf)
            h *= 2
        h = 2
        while h <= n:
            hf = h // 2
            for i in range(0, size[0], n):
                matrix[i:(i + n)] = self.t_col(matrix[i:(i + n)], n, h, hf)
            h *= 2

        return matrix

    @abstractmethod
    def it_2d(self, coef: np.ndarray, p: int):
        size = coef.shape
        n = 2 ** p
        h = n
        while h >= 2:
            hf = h // 2
            for i in range(0, size[0], n):
                coef[i:(i + n)] = self.it_col(coef[i:(i + n)], n, h, hf)
            h //= 2
        h = n
        while h >= 2:
            hf = h // 2
            for j in range(0, size[1], n):
                coef[:, j:(j + n)] = self.it_row(coef[:, j:(j + n)], n, h, hf)
            h //= 2
        return coef
