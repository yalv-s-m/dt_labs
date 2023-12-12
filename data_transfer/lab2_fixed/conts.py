# Хи-квадрат с числом степеней свободы k = 3
# и Рэлея с sigma**2 = 2.0 в промежутке [0; 5].

from scipy.special import gamma
from scipy.integrate import quad
import numpy as np


k = 3.0
left = 0
right = 5


def chi_sq_pdf(x, k):
    return (1 / ((2 ** (k/2)) * gamma(k / 2))) * (x ** (k/2 - 1)) * np.exp(-x/2) 


def rayleigh_pdf(x):
    return (x / 2) * np.exp(-x**2 / 4)


def kl_divergence(p, q, left, right):
    func = lambda x: p(x) * np.log(p(x) / q(x))
    res, _ = quad(func, left, right)
    return res


def entropy_X(x):
    px = chi_sq_pdf(x, k)
    return -px * np.log(px)


def entropy_XY(x):
    px = chi_sq_pdf(x, k)
    qx = rayleigh_pdf(x)
    return -px * np.log(qx)


kl_div_XY = kl_divergence(lambda x: chi_sq_pdf(x, k), lambda x: rayleigh_pdf(x), left, right)
kl_div_XX = kl_divergence(lambda x: chi_sq_pdf(x, k), lambda x: chi_sq_pdf(x, k), left, right)
kl_div_YX = kl_divergence(lambda x: rayleigh_pdf(x), lambda x: chi_sq_pdf(x, k), left, right)
H_X, _ = quad(entropy_X, left, right)
H_XY, _ = quad(entropy_XY, left, right)


print("###################################################################################")
print("ХИ-КВАДРАТ И РАСПРЕДЕЛЕНИЕ РЭЛЕЯ")
print("###################################################################################")
print(f"Дивергенция Кульбака-Лейблера D(fX || gY): {kl_div_XY}")
print(f"Дивергенция D(fX || fX): {kl_div_XX}")
print(f"Проверка неравенства D(fX || gY) != D(gY || fX): {kl_div_XY} != {kl_div_YX}")
print(f"Энтропия H(X): {H_X}")
print(f"Энтропия H(X, Y): {H_XY}")
print(f"Проверка тождества D(fX || gY) = H(X, Y) - H(X): {kl_div_XY} = {H_XY - H_X}")




    

