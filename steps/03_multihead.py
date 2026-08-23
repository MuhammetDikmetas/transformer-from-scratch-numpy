import numpy as np

rng = np.random.default_rng(0)

T, d_model, n_head = 5, 8, 4
d_head = d_model // n_head

X = rng.normal(size=(T, d_model))
Wq = rng.normal(size=(d_model, d_model)) * 0.1
Wk = rng.normal(size=(d_model, d_model)) * 0.1
Wv = rng.normal(size=(d_model, d_model)) * 0.1
Wo = rng.normal(size=(d_model, d_model)) * 0.1

G = rng.normal(size=(T, d_model))


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def split_heads(x):
    """(T, d_model) -> (n_head, T, d_head)"""
    return x.reshape(T, n_head, d_head).transpose(1, 0, 2)


def merge_heads(x):
    """(n_head, T, d_head) -> (T, d_model)"""
    return x.transpose(1, 0, 2).reshape(T, d_model)


def mha_forward(X, Wq, Wk, Wv, Wo):
    Q = split_heads(X @ Wq)
    K = split_heads(X @ Wk)
    V = split_heads(X @ Wv)

    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_head)

    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)

    A = softmax(scores)
    Y = A @ V

    concat = merge_heads(Y)
    out = concat @ Wo

    cache = {"X": X, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
             "Q": Q, "K": K, "V": V, "A": A, "concat": concat, "mask": mask}
    return out, cache


def mha_backward(dout, cache):
    X, Wq, Wk, Wv, Wo = (cache["X"], cache["Wq"], cache["Wk"],cache["Wv"], cache["Wo"])
    Q, K, V, A = cache["Q"], cache["K"], cache["V"], cache["A"]
    concat, mask = cache["concat"], cache["mask"]

    dWo = concat.T @ dout
    dconcat = dout @ Wo.T

    dY = split_heads(dconcat)

    dA = dY @ V.transpose(0, 2, 1)
    dV = A.transpose(0, 2, 1) @ dY

    dscores = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))
    dscores = np.where(mask, 0.0, dscores)
    dscores = dscores / np.sqrt(d_head)

    dQ = dscores @ K
    dK = dscores.transpose(0, 2, 1) @ Q

    dQm = merge_heads(dQ)
    dKm = merge_heads(dK)
    dVm = merge_heads(dV)

    dWq = X.T @ dQm
    dWk = X.T @ dKm
    dWv = X.T @ dVm

    dX = dQm @ Wq.T + dKm @ Wk.T + dVm @ Wv.T

    return dWq, dWk, dWv, dWo, dX


def loss_fn():
    out, cache = mha_forward(X, Wq, Wk, Wv, Wo)
    return np.sum(out * G), cache


def numerical_grad(param, eps=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]

        param[idx] = original + eps
        loss_plus, _ = loss_fn()

        param[idx] = original - eps
        loss_minus, _ = loss_fn()

        param[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


out, cache = mha_forward(X, Wq, Wk, Wv, Wo)
print("out shape:", out.shape)
print("kafa sayisi:", n_head, "| her kafa boyutu:", d_head)
print("\n1. kafanin attention agirliklari:")
print(np.round(cache["A"][0], 3))

loss, cache = loss_fn()
dWq, dWk, dWv, dWo, dX = mha_backward(G, cache)

print("\n--- gradient check ---")
print(f"Wq: {relative_error(dWq, numerical_grad(Wq)):.3e}")
print(f"Wk: {relative_error(dWk, numerical_grad(Wk)):.3e}")
print(f"Wv: {relative_error(dWv, numerical_grad(Wv)):.3e}")
print(f"Wo: {relative_error(dWo, numerical_grad(Wo)):.3e}")
print(f"X : {relative_error(dX,  numerical_grad(X)):.3e}")