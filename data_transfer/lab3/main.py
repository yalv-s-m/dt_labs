# Вариант #3

import numpy as np
import matplotlib.pyplot as plt


f0 = 100
fd0 = 30
f1 = 500
fd1 = 40
f2 = 10
Fmax = 20
N = 1000
fs = 10000
A = 0.1
Ad = 9


def U0(n):
    return A * np.sin((2 * np.pi * n * f0) / fs) * np.cos((2 * np.pi * fd0) / fs)


def U1(n):
    return A * np.sign(np.sin(((2 * np.pi * n * f1) / fs) + (Ad * np.cos((2 * np.pi * n * fd1) / fs))))


def U2(n):
    return A * np.cos(2 * np.pi * n * ((f2/fs) + ((Fmax - f2) * n) / (fs*N)))


A_matrix = np.array([[0.3352, -0.4895, -0.8027],
                     [-0.000019, -0.5279 , -0.9324],
                     [-0.3778, 0.3028, 0.1251]])


W_matrix = np.array([[-46.93, 19.67, -41.48],
                     [71.32, -60.21, 10.34],
                     [-58.82, 43.54, -52.20]])


W_matrix_correct_sign = np.array([[-46.93, 19.67, -41.48],
                                  [71.32, -60.21, 10.34],
                                  [58.82, -43.54, 52.20]])



samples = 200
U0_samples = np.asarray([U0(n) for n in range(samples)])
U1_samples = np.asarray([U1(n) for n in range(samples)])
U2_samples = np.asarray([U2(n) for n in range(samples)])
X = np.column_stack((U0_samples, U1_samples, U2_samples)).T
#print("Матрица Х:\n", X) 

  
mixed = A_matrix @ X
recovered = W_matrix_correct_sign @ mixed
Y = np.roll(recovered, -1, axis=0)


P = W_matrix_correct_sign @ A_matrix
def calc_J(P):
    m, _ = P.shape
    J = 0
    
    for i in range(m):
        elem1 = 0
        for j in range(m):
            elem1 += np.abs(P[i, j]) / np.max(np.abs(P[i, :]))
        J += (elem1 - 1)
            
    for j in range(m):
        elem2 = 0
        for i in range(m):
            elem2 += np.abs(P[i, j]) / np.max(np.abs(P[:, i]))
        J += (elem2 - 1)
 
    return J
    

J = calc_J(P)    
print(f"Глобальный индекс отклонения J = {J}")

X_normalized = X / np.max(np.abs(X), axis=1)[:, np.newaxis]
Y_normalized = Y / np.max(np.abs(Y), axis=1)[:, np.newaxis]
 

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(X_normalized[0], label="Входная")
plt.plot(Y_normalized[0], label="Восстановленная")
plt.title("Входная реализация X[0] и восстановленная Y[0]")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(X_normalized[1], label="Входная")
plt.plot(Y_normalized[1], label="Восстановленная")
plt.title("Входная реализация X[1] и восстановленная Y[1]")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(X_normalized[2], label="Входная")
plt.plot(Y_normalized[2], label="Восстановленная")
plt.title("Входная реализация X[2] и восстановленная Y[2]")
plt.legend()

plt.tight_layout()
plt.show()
