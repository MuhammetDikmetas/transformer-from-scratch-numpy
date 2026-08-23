import numpy as np

rng = np.random.default_rng(0)

T, d_model, d_head = 5, 8, 8

X = rng.normal(size=(T, d_model))
Wq = rng.normal(size=(d_model, d_head)) * 0.1
Wk = rng.normal(size=(d_model, d_head)) * 0.1
Wv = rng.normal(size=(d_model, d_head)) * 0.1

G = rng.normal(size=(T, d_head))


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def attention_forward(X, Wq, Wk, Wv):
    T = X.shape[0]
    dk = Wq.shape[1]

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    scores = (Q @ K.T) / np.sqrt(dk)

    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)

    A = softmax(scores)
    out = A @ V

    cache = {"X": X, "Wq": Wq, "Wk": Wk, "Wv": Wv,
             "Q": Q, "K": K, "V": V, "A": A, "mask": mask, "dk": dk}
    return out, cache


def attention_backward(dout, cache):
    X, Wq, Wk, Wv = cache["X"], cache["Wq"], cache["Wk"], cache["Wv"]
    Q, K, V, A = cache["Q"], cache["K"], cache["V"], cache["A"]
    mask, dk = cache["mask"], cache["dk"]

    dA = dout @ V.T
    dV = A.T @ dout

    dscores = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))
    dscores = np.where(mask, 0.0, dscores)
    dscores = dscores / np.sqrt(dk)

    dQ = dscores @ K
    dK = dscores.T @ Q

    dWq = X.T @ dQ
    dWk = X.T @ dK
    dWv = X.T @ dV

    dX = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T

    return dWq, dWk, dWv, dX


def loss_fn(X, Wq, Wk, Wv):
    out, cache = attention_forward(X, Wq, Wk, Wv)
    return np.sum(out * G), cache


def numerical_grad(param, eps=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]

        param[idx] = original + eps
        loss_plus, _ = loss_fn(X, Wq, Wk, Wv)

        param[idx] = original - eps
        loss_minus, _ = loss_fn(X, Wq, Wk, Wv)

        param[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


out, cache = attention_forward(X, Wq, Wk, Wv)
print("out shape:", out.shape)
print("\nattention agirliklari (causal mask):")
print(np.round(cache["A"], 3))

loss, cache = loss_fn(X, Wq, Wk, Wv)
dWq, dWk, dWv, dX = attention_backward(G, cache)

print("\n--- gradient check ---")
print(f"Wq: {relative_error(dWq, numerical_grad(Wq)):.3e}")
print(f"Wk: {relative_error(dWk, numerical_grad(Wk)):.3e}")
print(f"Wv: {relative_error(dWv, numerical_grad(Wv)):.3e}")
print(f"X : {relative_error(dX,  numerical_grad(X)):.3e}")