import numpy as np

rng = np.random.default_rng(0)

T, d_head = 5, 8

X = rng.normal(size=(T, d_head))
G = rng.normal(size=(T, d_head))


def rope_tables(T, d, base=10000.0):
    i = np.arange(d // 2)
    theta = 1.0 / (base ** (2.0 * i / d))
    pos = np.arange(T)
    angles = pos[:, None] * theta[None, :]
    return np.cos(angles), np.sin(angles)


def rope_forward(X, cos, sin):
    x1 = X[:, 0::2]
    x2 = X[:, 1::2]

    out = np.empty_like(X)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos

    cache = {"cos": cos, "sin": sin}
    return out, cache


def rope_backward(dout, cache):
    cos, sin = cache["cos"], cache["sin"]

    d1 = dout[:, 0::2]
    d2 = dout[:, 1::2]

    dX = np.empty_like(dout)
    dX[:, 0::2] = d1 * cos + d2 * sin
    dX[:, 1::2] = -d1 * sin + d2 * cos

    return dX


cos, sin = rope_tables(T, d_head)
out, cache = rope_forward(X, cos, sin)

print("out shape:", out.shape)
print("\ngirdi norm :", np.round(np.linalg.norm(X, axis=-1), 4))
print("cikti norm :", np.round(np.linalg.norm(out, axis=-1), 4))

dX = rope_backward(G, cache)
print("\ndX:", dX.shape)