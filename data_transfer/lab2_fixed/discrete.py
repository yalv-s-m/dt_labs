# Геометрическое распределение с p = 0.2 
# и Юла-Саймона при alpha = 5

import numpy as np
from scipy.stats import yulesimon
from scipy.special import rel_entr


p = 0.2

alpha = 5.0


def geom(x, p):
    return ((1 - p) ** x) * p


def kl_divergence(px, qx):
    return  px * np.log(px / qx)


def entropy_X(px):
    return -px * np.log(px)


def entropy_XY(px, qx):
    return -px * np.log(qx)


div_XY = 0
div_XY_scp = 0
div_XX = 0
div_YX = 0
H_X = 0
H_XY = 0


for val in range(0, 20):
    px_val = geom(val, p)
    qx_val = yulesimon.pmf(val, alpha)
    if qx_val == 0:
        continue

    div_XY += kl_divergence(px_val, qx_val)
    div_XY_scp += rel_entr(px_val, qx_val)
    div_XX += kl_divergence(px_val, px_val)
    div_YX += kl_divergence(qx_val, px_val)
    H_X += entropy_X(px_val)
    H_XY += entropy_XY(px_val, qx_val)
    

print("###################################################################################")
print("ГЕОМЕТРИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ И РАСПРЕДЕЛЕНИЕ ЮЛА-САЙМОНА")
print("###################################################################################")
print(f"Дивергенция(вручную) D(fX || gY): {div_XY}")
print(f"Дивергенция через scipy.special.rel_entr: {div_XY_scp}")
print(f"Дивергенция D(fX || fX): {div_XX}")
print(f"Проверка неравенства D(fX || gY) != D(gY || fX): {div_XY} != {div_YX}")
print(f"Энтропия H(X): {H_X}")
print(f"Энтропия H(X, Y): {H_XY}")
print(f"Проверка тождества D(fX || gY) = H(X, Y) - H(X): {div_XY} = {H_XY - H_X}")




