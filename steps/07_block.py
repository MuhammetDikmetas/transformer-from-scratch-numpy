import numpy as np

rng = np.random.default_rng(0)

T, d_model, n_head = 6, 8, 4
d_head = d_model // n_head
d_ff = 16

X = rng.normal(size=(T, d_model))
G = rng.normal(size=(T, d_model))

g1 = np.ones(d_model)
Wq = rng.normal(size=(d_model, d_model)) * 0.1
Wk = rng.normal(size=(d_model, d_model)) * 0.1
Wv = rng.normal(size=(d_model, d_model)) * 0.1
Wo = rng.normal(size=(d_model, d_model)) * 0.1

g2 = np.ones(d_model)
Wg = rng.normal(size=(d_model, d_ff)) * 0.1
Wu = rng.normal(size=(d_model, d_ff)) * 0.1
Wd = rng.normal(size=(d_ff, d_model)) * 0.1


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def split_heads(x):
    return x.reshape(T, n_head, d_head).transpose(1, 0, 2)


def merge_heads(x):
    return x.transpose(1, 0, 2).reshape(T, d_model)


def rope_tables(T, d, base=10000.0):
    i = np.arange(d // 2)
    theta = 1.0 / (base ** (2.0 * i / d))
    pos = np.arange(T)
    angles = pos[:, None] * theta[None, :]
    return np.cos(angles), np.sin(angles)


def rope_apply(x, cos, sin):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def rope_apply_backward(dout, cos, sin):
    d1 = dout[..., 0::2]
    d2 = dout[..., 1::2]
    dx = np.empty_like(dout)
    dx[..., 0::2] = d1 * cos + d2 * sin
    dx[..., 1::2] = -d1 * sin + d2 * cos
    return dx


def rmsnorm_forward(X, g, eps=1e-6):
    rms = np.sqrt(np.mean(X ** 2, axis=-1, keepdims=True) + eps)
    xhat = X / rms
    return xhat * g, {"xhat": xhat, "rms": rms, "g": g}


def rmsnorm_backward(dout, cache):
    xhat, rms, g = cache["xhat"], cache["rms"], cache["g"]
    dg = np.sum(dout * xhat, axis=0)
    dxhat = dout * g
    dX = (dxhat - xhat * np.mean(dxhat * xhat, axis=-1, keepdims=True)) / rms
    return dX, dg


def mha_forward(X, Wq, Wk, Wv, Wo, cos, sin):
    Q = rope_apply(split_heads(X @ Wq), cos, sin)
    K = rope_apply(split_heads(X @ Wk), cos, sin)
    V = split_heads(X @ Wv)

    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_head)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)

    A = softmax(scores)
    concat = merge_heads(A @ V)
    out = concat @ Wo

    cache = {"X": X, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
             "Q": Q, "K": K, "V": V, "A": A, "concat": concat,
             "mask": mask, "cos": cos, "sin": sin}
    return out, cache


def mha_backward(dout, cache):
    X, Wq, Wk, Wv, Wo = (cache["X"], cache["Wq"], cache["Wk"],
                         cache["Wv"], cache["Wo"])
    Q, K, V, A = cache["Q"], cache["K"], cache["V"], cache["A"]
    concat, mask = cache["concat"], cache["mask"]
    cos, sin = cache["cos"], cache["sin"]

    dWo = concat.T @ dout
    dY = split_heads(dout @ Wo.T)

    dA = dY @ V.transpose(0, 2, 1)
    dV = A.transpose(0, 2, 1) @ dY

    dscores = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))
    dscores = np.where(mask, 0.0, dscores) / np.sqrt(d_head)

    dQ = rope_apply_backward(dscores @ K, cos, sin)
    dK = rope_apply_backward(dscores.transpose(0, 2, 1) @ Q, cos, sin)

    dQm = merge_heads(dQ)
    dKm = merge_heads(dK)
    dVm = merge_heads(dV)

    dWq = X.T @ dQm
    dWk = X.T @ dKm
    dWv = X.T @ dVm
    dX = dQm @ Wq.T + dKm @ Wk.T + dVm @ Wv.T

    return dX, dWq, dWk, dWv, dWo


def swiglu_forward(X, Wg, Wu, Wd):
    gate = X @ Wg
    up = X @ Wu
    s = sigmoid(gate)
    silu = gate * s
    h = silu * up
    out = h @ Wd
    cache = {"X": X, "Wg": Wg, "Wu": Wu, "Wd": Wd,
             "gate": gate, "up": up, "s": s, "silu": silu, "h": h}
    return out, cache


def swiglu_backward(dout, cache):
    X, Wg, Wu, Wd = cache["X"], cache["Wg"], cache["Wu"], cache["Wd"]
    gate, up, s, silu, h = (cache["gate"], cache["up"], cache["s"],
                            cache["silu"], cache["h"])

    dWd = h.T @ dout
    dh = dout @ Wd.T

    dsilu = dh * up
    dup = dh * silu
    dgate = dsilu * (s * (1.0 + gate * (1.0 - s)))

    dWg = X.T @ dgate
    dWu = X.T @ dup
    dX = dgate @ Wg.T + dup @ Wu.T

    return dX, dWg, dWu, dWd


def block_forward(X, params, cos, sin):
    g1, Wq, Wk, Wv, Wo, g2, Wg, Wu, Wd = params

    n1, c_n1 = rmsnorm_forward(X, g1)
    att, c_att = mha_forward(n1, Wq, Wk, Wv, Wo, cos, sin)
    h = X + att

    n2, c_n2 = rmsnorm_forward(h, g2)
    ff, c_ff = swiglu_forward(n2, Wg, Wu, Wd)
    out = h + ff

    cache = {"c_n1": c_n1, "c_att": c_att, "c_n2": c_n2, "c_ff": c_ff}
    return out, cache


def block_backward(dout, cache):
    c_n1, c_att, c_n2, c_ff = (cache["c_n1"], cache["c_att"],
                               cache["c_n2"], cache["c_ff"])

    dff = dout
    dn2, dWg, dWu, dWd = swiglu_backward(dff, c_ff)
    dh_norm, dg2 = rmsnorm_backward(dn2, c_n2)
    dh = dout + dh_norm

    datt = dh
    dn1, dWq, dWk, dWv, dWo = mha_backward(datt, c_att)
    dX_norm, dg1 = rmsnorm_backward(dn1, c_n1)
    dX = dh + dX_norm

    return dX, (dg1, dWq, dWk, dWv, dWo, dg2, dWg, dWu, dWd)


cos, sin = rope_tables(T, d_head)
params = (g1, Wq, Wk, Wv, Wo, g2, Wg, Wu, Wd)

out, cache = block_forward(X, params, cos, sin)
print("girdi shape:", X.shape, "| cikti shape:", out.shape)

dX, grads = block_backward(G, cache)
names = ["g1", "Wq", "Wk", "Wv", "Wo", "g2", "Wg", "Wu", "Wd"]
print("\ngradyanlar:")
for name, gr, p in zip(names, grads, params):
    print(f"  {name}: {gr.shape}  (parametre: {p.shape})")
print("\ndX:", dX.shape)