from modules.algorithms.algorithms import FastAlgorithm
import numpy as np


class FWT(FastAlgorithm):

    def t_row(self, lam: np.ndarray, n: int, h: int, hf: int):
        for i in range(0, n, h):
            for j in range(hf):
                u, v = lam[:, i + j], lam[:, i + j + hf]
                lam[:, i + j], lam[:, i + j + hf] = (u + v) / 2, (u - v) / 2
        return lam

    def t_col(self, lam: np.ndarray, n: int, h: int, hf: int):
        for i in range(0, n, h):
            for j in range(hf):
                u, v = lam[i + j], lam[i + j + hf]
                lam[i + j], lam[i + j + hf] = (u + v) / 2, (u - v) / 2
        return lam

    def it_row(self, coef: np.ndarray, n: int, h: int, hf: int):
        for i in range(0, n, h):
            for j in range(hf):
                u, v = coef[:, i + j], coef[:, i + j + hf]
                coef[:, i + j], coef[:, i + j + hf] = (u + v), (u - v)
        return coef

    def it_col(self, coef: np.ndarray, n: int, h: int, hf: int):
        for i in range(0, n, h):
            for j in range(hf):
                u, v = coef[i + j], coef[i + j + hf]
                coef[i + j], coef[i + j + hf] = (u + v), (u - v)
        return coef

    def t_2d(self, matrix: np.ndarray, p=3):
        return super().t_2d(matrix, p)

    def it_2d(self, coef: np.ndarray, p=3):
        return super().it_2d(coef, p)
