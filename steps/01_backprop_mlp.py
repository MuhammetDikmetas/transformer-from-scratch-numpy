import numpy as np

rng = np.random.default_rng(42)

# N → number of samples   → 20
# D → input dimension     → 2
# H → hidden size         → 8
# C → number of classes   → 3
N, D, H, C = 20, 2, 8, 3

X = rng.normal(size=(N, D))
y = rng.integers(0, C, size=N)

W1 = rng.normal(size=(D, H)) * 0.1
b1 = np.zeros(H)
W2 = rng.normal(size=(H, C)) * 0.1
b2 = np.zeros(C)


def softmax(z):
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X, y, W1, b1, W2, b2):
    n = X.shape[0]
    z1 = X @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    p = softmax(z2)
    loss = np.mean(-np.log(p[np.arange(n), y] + 1e-12))
    cache = {"X": X, "y": y, "a1": a1, "p": p, "W2": W2}
    return loss, cache


def backward(cache):
    X, y, a1, p, W2 = cache["X"], cache["y"], cache["a1"], cache["p"], cache["W2"]
    n = X.shape[0]

    dz2 = p.copy()
    dz2[np.arange(n), y] -= 1.0
    dz2 /= n

    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * (1.0 - a1 ** 2)

    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return dW1, db1, dW2, db2


def numerical_grad(param, eps=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]

        param[idx] = original + eps
        loss_plus, _ = forward(X, y, W1, b1, W2, b2)

        param[idx] = original - eps
        loss_minus, _ = forward(X, y, W1, b1, W2, b2)

        param[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


loss, cache = forward(X, y, W1, b1, W2, b2)
print("baslangic loss:", round(loss, 4))

dW1, db1, dW2, db2 = backward(cache)

print("\n--- gradient check ---")
print("W1:", relative_error(dW1, numerical_grad(W1)))
print("b1:", relative_error(db1, numerical_grad(b1)))
print("W2:", relative_error(dW2, numerical_grad(W2)))
print("b2:", relative_error(db2, numerical_grad(b2)))

print("\n--- egitim ---")
lr = 0.5
for step in range(1001):
    loss, cache = forward(X, y, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward(cache)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss:.4f}")