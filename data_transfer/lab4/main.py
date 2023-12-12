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


def sigm(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigm(x):
    return np.exp(-x) / (1 + np.exp(-x))**2


A_matrix = np.array([[0.3352, -0.4895, -0.8027],
                     [-0.000019, -0.5279 , -0.9324],
                     [-0.3778, 0.3028, 0.1251]])


'''
W_matrix_correct_sign = np.array([[-46.93, 19.67, -41.48],
                                  [71.32, -60.21, 10.34],
                                  [58.82, -43.54, 52.20]])
'''

samples = 800
U0_samples = np.asarray([U0(n) for n in range(samples)])
U1_samples = np.asarray([U1(n) for n in range(samples)])
U2_samples = np.asarray([U2(n) for n in range(samples)])
X = np.column_stack((U0_samples, U1_samples, U2_samples)).T


mixed = A_matrix @ X

# Центрирование 
centered_X = X - np.mean(X, axis=1, keepdims=True)


# "Отбеливание" (whitening)
cov_X = np.cov(centered_X)
D, E = np.linalg.eigh(cov_X)
D_inverted = np.diag(1.0 / np.sqrt(D))

# x = E(D^-1/2)(E^T)x
whitened_X = E @ D_inverted @ E.T @ centered_X


n, _ = count_src, _ = X.shape
np.random.seed(1)
W = np.random.rand(n, count_src)
#W /= np.linalg.norm(), axis=1)[:, np.newaxis]
for i in range(count_src):
    w = np.random.rand(n)
    for _ in range(1000):
        # w^T @ X
        current_signal_proj = w.T @ whitened_X
        # g(w^T @ X)
        sigm_val = sigm(current_signal_proj)  
        # g'(w^T @ X)
        sigm_val_deriv = deriv_sigm(current_signal_proj)
        
        # Let w^+ = E{Xg(w^T * X)} - E{g'(w^T * X)} * w
        # E - мат. ожидание
        w_plus = np.mean(whitened_X * sigm_val, axis=1) - np.mean(sigm_val_deriv) * w
        
        # Let w = w^+ / ||w^+||
        w = w_plus / np.linalg.norm(w_plus)
        
        # Ортогонолизация Грама-Шмидта
        for j in range(i):
            w = w - np.sum(w.T * W[:, j]) * W[:, j]
            w = w / np.linalg.norm(w)
                
    W[:, i] = w
restored = W.T @ whitened_X


X_normalized = X / np.max(np.abs(X), axis=1)[:, np.newaxis]
Y_normalized = restored / np.max(np.abs(restored), axis=1)[:, np.newaxis]

Y_normalized = np.roll(Y_normalized, shift=-1, axis=0) 

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(X_normalized[0], label="Входная")
plt.plot(Y_normalized[1], label="Восстановленная")
plt.title("Входная реализация X[0] и восстановленная Y[0]")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(X_normalized[1], label="Входная")
plt.plot(Y_normalized[0], label="Восстановленная")
plt.title("Входная реализация X[1] и восстановленная Y[1]")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(X_normalized[2], label="Входная")
plt.plot(Y_normalized[2], label="Восстановленная")
plt.title("Входная реализация X[2] и восстановленная Y[2]")
plt.legend()

plt.tight_layout()
plt.show()