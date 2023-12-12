# Вариант №3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math


lambda_val = 0.5
size = 10 ** 4


def generate_rand_exp_nums(lambda_val: float, size :int) -> list:
    seed = 30
    randoms = []
    
    for _ in range(size):
        # Генерируем случайное число от 0 до 1
        seed = (16807 * seed + 2147483647) & 0xFFFFFFFF
        random_float = seed / 0xFFFFFFFF
        
        # Применяем обратную функцию экспоненциального распределения
        random_exp = -math.log(1 - random_float) / lambda_val
        
        randoms.append(random_exp)
    
    return randoms


sample = generate_rand_exp_nums(lambda_val, size)
plt.hist(sample, bins = 30, density=True, alpha = 0.6, label='Гистограмма')


x_axis_values = np.linspace(0, max(sample), 1000)
func = lambda_val * np.exp(-lambda_val * x_axis_values)
plt.plot(x_axis_values, func, 'r', lw=2, label='Аналитическая зависимость')


x_kde = np.linspace(0, max(sample), 1000)
kde = gaussian_kde(sample, bw_method = 0.01) # Kernel Density Estimation
plt.plot(x_kde, kde(x_kde), 'g', lw=2, label='Ядерная оценка плотности')


plt.legend()
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.show()


